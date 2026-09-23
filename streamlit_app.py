import json
import os
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.config.config import Config, moderation
from src.config.usage import get_usage_gate
from src.document_ingestion.document_processor import DocumentHandler
from src.document_ingestion.document_processor import normalize_source_key
from src.document_ingestion.document_processor import relabel_sources
from src.graph_builder.graph import GraphBuilder
from src.suggestions import deterministic_followups, generate_followups
from src.vector_store.store import VectorStore

load_dotenv()


def _secrets_get(key):
    """Read a Streamlit secret; None when not deployed (no secrets.toml).

    st.secrets raises if no secrets.toml exists (local dev) — treat that
    as "no value" rather than a failure.
    """
    try:
        return st.secrets.get(key)
    except Exception:
        return None


DEPLOYED = bool(_secrets_get("DEPLOYED"))


def _bridge_secret(secret_key, env_var):
    """Lift a server-side secret into the environment (deployed mode only)."""
    value = _secrets_get(secret_key)
    if value:
        os.environ[env_var] = value
        return True
    return False


# Allow deployed instances to override the default agent system prompt from
# Streamlit secrets without touching the repo (see prompts/agent_v1.md).
prompt_override = _secrets_get("AGENT_SYSTEM_PROMPT")
if prompt_override:
    os.environ["AGENT_SYSTEM_PROMPT"] = prompt_override

# API keys live server-side; .env / sidebar inputs still apply locally.
_bridge_secret("OPENAI_API_KEY", "OPENAI_API_KEY")
_bridge_secret("TAVILY_API_KEY", "TAVILY_API_KEY")

st.set_page_config(page_title="NoteForge RAG")
st.title("NoteForge — RAG Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "followups" not in st.session_state:
    st.session_state.followups = {}
if "nf_session_id" not in st.session_state:
    st.session_state.nf_session_id = uuid.uuid4().hex

# Widget state (e.g. the URL input) can only be changed BEFORE the widget is
# instantiated in a run, so the "clear the URL box" request is carried via a
# plain flag and applied here while the sidebar widgets don't exist yet.
if st.session_state.pop("nf_clear_url", False):
    st.session_state["nf_url_input"] = ""

RATE_LIMIT_MSG = (
    "You've hit the demo's temporary usage limit — please wait a minute "
    "and try again."
)
ERROR_MSG = "Sorry, something went wrong while generating that answer. Please try again."

# Process-wide budget keeper (module import is cached per process, so this
# singleton — and the global caps inside it — survive Streamlit reruns).
g_usage = get_usage_gate()

# Memoized demo ingest: the data/ corpus is embedded once per process and
# reused across sessions so page reloads don't re-burn embedding tokens.
_INGEST_CACHE: dict = {}


def render_copy_button(text: str) -> None:
    """Small HTML component that copies `text` to the clipboard on click.

    Streamlit strips <script> from markdown, so a components.html iframe is
    used (real JS). execCommand('copy') + hidden textarea works inside the
    sandboxed iframe without clipboard-permission issues. The answer is
    embedded as a JSON literal, so quotes/newlines/markdown are safe.
    """
    payload = json.dumps(text)
    components.html(
        "<button id='nf-copy' onclick='copy()'>Copy</button>"
        "<style>"
        "button{border:1px solid #ccc;border-radius:6px;background:#fff;"
        "font-family:system-ui;font-size:13px;padding:2px 10px;cursor:pointer}"
        "button:active{background:#e6e6e6}"
        "</style>"
        "<script>"
        "function copy(){"
        f"var t={payload};"
        "var ta=document.createElement('textarea');"
        "ta.value=t;"
        "document.body.appendChild(ta);"
        "ta.select();"
        "try{document.execCommand('copy');}catch(e){}"
        "document.body.removeChild(ta);"
        "var b=document.getElementById('nf-copy');"
        "b.textContent='Copied!';"
        "setTimeout(function(){b.textContent='Copy';},1500);"
        "}"
        "</script>",
        height=40,
        width=90,
    )


def clear_session_state():
    for key in [
        "graph",
        "retriever",
        "source_files",
        "paper_metadata",
        "chunk_counts",
        "messages",
        "followups",
        "nf_generating",
        "nf_question",
    ]:
        st.session_state.pop(key, None)
    st.session_state.messages = []
    st.session_state.followups = {}


def _stream_chunks(text: str, max_updates: int = 60) -> list[str]:
    """Split text into ~`max_updates` progressively longer prefixes for a
    cosmetic streaming reveal. Empty/very short inputs collapse to one step."""
    text = text or ""
    if len(text) <= 8:
        return [text]
    n = max(1, min(max_updates, len(text)))
    step = max(1, (len(text) + n - 1) // n)
    return [text[:i] for i in range(step, len(text) + step, step)]


def _stream_markdown(text: str) -> None:
    """Reveal `text` into a live placeholder, Perplexity-style (cosmetic)."""
    ph = st.empty()
    prefixes = _stream_chunks(text)
    if len(prefixes) <= 1:
        ph.markdown(text)
        return
    delay = max(0.008, min(0.08, 3.0 / len(prefixes)))
    for i, prefix in enumerate(prefixes):
        ph.markdown(prefix + ("▌" if i < len(prefixes) - 1 else ""))
        time.sleep(delay)


def _answer_question(question: str):
    """Run the pipeline for `question`. Returns
    (answer, task_type, task_label, result); failures degrade to ERROR_MSG."""
    try:
        result = st.session_state.graph.run(
            question,
            source_files=st.session_state.get("source_files", []),
            paper_metadata=st.session_state.get("paper_metadata", {}),
            chat_history=st.session_state.messages[:-1],
        )
        answer = result.get("answer", "No answer generated.")
        task_type = result.get("task_type", "qa")
        task_label = "agent" if task_type == "qa" else "synthesis"
    except Exception:
        answer = ERROR_MSG
        task_type = "qa"
        task_label = None
        result = {}
    return answer, task_type, task_label, result


def _reveal_followups(idx, task_type, task_label, result) -> None:
    """Compute + reveal follow-up chips inside the answer bubble (same run).
    LLM-generated when the budget allows, else deterministic templates —
    never a hard failure path."""
    if not task_label:
        return
    ph = st.empty()
    ph.caption("Suggesting follow-ups…")
    if g_usage.try_llm(st.session_state.nf_session_id):
        followups = generate_followups(
            result,
            Config.get_llm(),
            st.session_state.get("source_files", []),
            st.session_state.get("paper_metadata", {}),
        )
    else:
        followups = deterministic_followups(
            st.session_state.get("source_files", []),
            st.session_state.get("paper_metadata", {}),
            task_type,
        )
    ph.empty()
    for j, q in enumerate(followups):
        if st.button(f"Ask: {q}", key=f"fup_{idx}_{j}"):
            st.session_state.pending_followup = q
            st.rerun()
        time.sleep(0.12)
    st.session_state.followups[idx] = followups


def _demo_cache_key(pdfs, url):
    """Cache key for the auto-ingest demo path (data/ only, no URL)."""
    if not pdfs or url:
        return None
    files = tuple(sorted(p.name for p in pdfs))
    return (files, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP, Config.RETRIEVER_K)


def ingest_documents(uploaded_files=None, url=None):
    handler = DocumentHandler()

    if uploaded_files:
        pdfs = uploaded_files
        source_from_upload = True
    else:
        data_dir = Path("data")
        if not data_dir.is_dir() and not url:
            st.info("Upload PDFs or add a URL using the sidebar to get started.")
            return
        pdfs = sorted(data_dir.glob("*.pdf")) + sorted(data_dir.glob("*.txt"))
        if not pdfs and not url:
            st.info("Upload PDFs or add a URL using the sidebar to get started.")
            return
        source_from_upload = False

    # Demo/data path only: serve from the per-process cache when present so
    # reloads and repeat visits don't re-embed the corpus.
    cache_key = None if (source_from_upload or url) else _demo_cache_key(pdfs, url)
    if cache_key is not None and cache_key in _INGEST_CACHE:
        (
            st.session_state.retriever,
            st.session_state.source_files,
            st.session_state.paper_metadata,
            st.session_state.chunk_counts,
            st.session_state.graph,
        ) = _INGEST_CACHE[cache_key]
        with st.status("Documents ready", expanded=False) as status:
            status.update(
                label="Ready — loaded from in-process cache (no re-embedding)",
                state="complete",
            )
        return

    if not g_usage.try_ingest():
        st.warning(
            "Demo usage limit for document processing reached — "
            "please try again later."
        )
        return

    try:
        all_docs = []
        temp_paths = []

        with st.status("Loading documents…", expanded=True) as status:
            for pdf in pdfs:
                if source_from_upload:
                    suffix = Path(pdf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(pdf.getvalue())
                        temp_path = tmp.name
                    temp_paths.append(temp_path)
                    file_label = pdf.name
                else:
                    temp_path = str(pdf)
                    file_label = pdf.name

                status.write(f"Loading {file_label} …")
                suffix = Path(file_label).suffix
                loaded = (
                    handler.pdf_loader(temp_path)
                    if suffix == ".pdf"
                    else handler.text_loader(temp_path)
                )
                if source_from_upload:
                    relabel_sources(loaded, file_label)
                all_docs.extend(loaded)

            if url:
                status.write(f"Loading {url} …")
                web_docs = handler.url_loader(url)
                for doc in web_docs:
                    doc.metadata["source"] = url
                all_docs.extend(web_docs)

            if source_from_upload:
                status.write("Cleaning up temporary files …")
                for path in temp_paths:
                    os.unlink(path)

            status.write("Splitting into chunks …")
            chunks = handler.doc_splitter(all_docs)

            source_counts = Counter(
                doc.metadata.get("source", "unknown") for doc in chunks
            )

            status.write(f"Embedding {len(chunks)} chunks …")
            vs = VectorStore()
            vs.create_retriever(chunks)
            retriever = vs.get_retriever()

            filenames = [Path(pdf.name).name for pdf in pdfs]
            if url:
                filenames.append(normalize_source_key(url))

            status.write("Extracting structured paper metadata …")
            llm = Config.get_llm()
            paper_metadata = handler.extract_metadata(all_docs, llm)

            status.write("Building graph …")
            builder = GraphBuilder(retriever, llm)
            builder.build()

            status.update(
                label=f"Ready — {len(filenames)} sources, {len(chunks)} chunks",
                state="complete",
            )
    except Exception:
        st.error(ERROR_MSG)
        return

    st.session_state.retriever = retriever
    st.session_state.source_files = filenames
    st.session_state.paper_metadata = paper_metadata
    st.session_state.chunk_counts = source_counts
    st.session_state.graph = builder

    if cache_key is not None:
        _INGEST_CACHE[cache_key] = (
            retriever,
            filenames,
            paper_metadata,
            source_counts,
            builder,
        )


def render_message(role, content, task=None):
    """Render a chat message with a task caption + copy button for replies.

    Copy buttons are rendered ONLY from here (the message loop). Keyless
    components.html iframes are matched by script position, so rendering a
    copy button anywhere else (e.g. inline in the handler) made the iframe
    re-mount across runs — the transient duplicate that flashed over the
    question. One render site keeps every iframe in a stable position.
    """
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            if task:
                st.caption(f"task: {task}")
            render_copy_button(content)


with st.sidebar:
    if DEPLOYED:
        st.caption(
            "Demo instance — API keys are configured server-side and never "
            "shown here."
        )
        api_key = os.getenv("OPENAI_API_KEY", "")
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    else:
        st.header("API Keys")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        tavily_api_key = st.text_input(
            "Tavily API Key",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
        )
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    url_input = st.text_input(
        "Or add a URL",
        placeholder="https://…",
        key="nf_url_input",
        help="Treat a web page as a first-class source alongside PDFs.",
    )

    if (uploaded_files or url_input) and st.button(
        "Process Documents", type="primary"
    ):
        clear_session_state()
        ingest_documents(uploaded_files, url=url_input or None)
        if "graph" in st.session_state:
            st.session_state.nf_clear_url = True
            st.rerun()

    st.divider()

    if (
        not DEPLOYED
        and api_key
        and "graph" not in st.session_state
        and not uploaded_files
        and not url_input
    ):
        ingest_documents()

    if "chunk_counts" in st.session_state:
        st.caption("Index")
        for source, count in st.session_state.chunk_counts.most_common():
            st.caption(f"  {Path(source).name} — {count} chunks")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.followups = {}
        st.session_state.pop("nf_generating", None)
        st.session_state.pop("nf_question", None)

if DEPLOYED and (
        "graph" not in st.session_state or st.session_state.graph is None
    ):
    st.info(
        "No documents loaded. Upload PDFs or add a URL from the sidebar to "
        "get started."
    )

# Conversations live only in this browser session. A native beforeunload
# confirm isn't possible inside Streamlit's sandboxed components.html
# iframes, so an in-app banner is the honest equivalent.
if st.session_state.messages:
    st.warning(
        "This session keeps your conversation only while this tab is open — "
        "refreshing clears it."
    )

for i, msg in enumerate(st.session_state.messages):
    render_message(msg["role"], msg["content"], msg.get("task"))
    if msg["role"] == "assistant" and i in st.session_state.followups:
        for j, q in enumerate(st.session_state.followups[i]):
            if st.button(f"Ask: {q}", key=f"fup_{i}_{j}"):
                st.session_state.pending_followup = q
                st.rerun()

# --- Answer-production run: the input is disabled while the graph works. ---
# Two-phase handler: the capture run below stashes the question and reruns
# with nf_generating set; THIS run streams the answer, reveals the follow-up
# chips in the same bubble, then reruns once more so the loop renders the
# message (caption + copy button + chips) and the input comes back enabled.
generating = st.session_state.pop("nf_generating", False)
if generating:
    question = st.session_state.pop("nf_question", "")
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            answer, task_type, task_label, result = _answer_question(question)
        _stream_markdown(answer)
        if task_label:
            st.caption(f"task: {task_type} → {task_label}")
        _reveal_followups(
            len(st.session_state.messages), task_type, task_label, result
        )
    assistant_msg = {"role": "assistant", "content": answer}
    if task_label:
        assistant_msg["task"] = f"{task_type} → {task_label}"
    st.session_state.messages.append(assistant_msg)

prompt = st.chat_input("Ask about your documents …", disabled=generating)

if generating:
    st.rerun()

pending = st.session_state.pop("pending_followup", None)
if pending:
    prompt = pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "graph" not in st.session_state or st.session_state.graph is None:
        answer = (
            "No documents loaded. Upload PDFs or add a URL using the sidebar."
        )
        task_label = None
        result = {}
        task_type = "qa"
    elif not g_usage.try_llm(st.session_state.nf_session_id):
        answer = RATE_LIMIT_MSG
        task_label = None
        result = {}
        task_type = "qa"
        with st.chat_message("assistant"):
            st.markdown(answer)
    elif not g_usage.try_moderation():
        answer = RATE_LIMIT_MSG
        task_label = None
        result = {}
        task_type = "qa"
        with st.chat_message("assistant"):
            st.markdown(answer)
    elif not moderation(prompt):
        answer = "I'm sorry, I can't help with that request."
        task_label = None
        result = {}
        task_type = "qa"
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        # Real question: defer to the generating run above (disabled input +
        # streamed answer), then rerun to complete.
        st.session_state["nf_question"] = prompt
        st.session_state["nf_generating"] = True
        st.rerun()

    st.session_state.messages.append({"role": "assistant", "content": answer})
