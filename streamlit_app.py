import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.config.config import Config, moderation
from src.document_ingestion.document_processor import DocumentHandler
from src.document_ingestion.document_processor import normalize_source_key
from src.graph_builder.graph import GraphBuilder
from src.vector_store.store import VectorStore

load_dotenv()

# Allow deployed instances to override the default agent system prompt from
# Streamlit secrets without touching the repo (see prompts/agent_v1.md).
# st.secrets raises if no secrets.toml exists (local dev) — treat that as no override.
try:
    prompt_override = st.secrets["AGENT_SYSTEM_PROMPT"]
except Exception:
    prompt_override = ""
if prompt_override:
    os.environ["AGENT_SYSTEM_PROMPT"] = prompt_override

st.set_page_config(page_title="NoteForge RAG")
st.title("NoteForge — RAG Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []


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
    ]:
        st.session_state.pop(key, None)
        st.session_state.messages = []


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

        source_counts = Counter(doc.metadata.get("source", "unknown") for doc in chunks)

        status.write(f"Embedding {len(chunks)} chunks …")
        vs = VectorStore()
        vs.create_retriever(chunks)
        st.session_state.retriever = vs.get_retriever()

        filenames = [Path(pdf.name).name for pdf in pdfs]
        if url:
            filenames.append(normalize_source_key(url))
        st.session_state.source_files = filenames

        status.write("Extracting structured paper metadata …")
        llm = Config.get_llm()
        st.session_state.paper_metadata = handler.extract_metadata(all_docs, llm)

        status.write("Building graph …")
        builder = GraphBuilder(st.session_state.retriever, llm)
        builder.build()
        st.session_state.graph = builder
        st.session_state.chunk_counts = source_counts

        status.update(
            label=f"Ready — {len(filenames)} sources, {len(chunks)} chunks",
            state="complete",
        )


def render_message(role, content):
    """Render a chat message with a copy button for assistant replies."""
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            render_copy_button(content)


with st.sidebar:
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
        help="Treat a web page as a first-class source alongside PDFs.",
    )

    if (uploaded_files or url_input) and st.button(
        "Process Documents", type="primary"
    ):
        clear_session_state()
        ingest_documents(uploaded_files, url=url_input or None)

    st.divider()

    if api_key and "graph" not in st.session_state and not uploaded_files and not url_input:
        ingest_documents()

    if "chunk_counts" in st.session_state:
        st.caption("Index")
        for source, count in st.session_state.chunk_counts.most_common():
            st.caption(f"  {Path(source).name} — {count} chunks")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    render_message(msg["role"], msg["content"])

if prompt := st.chat_input("Ask about your documents …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "graph" not in st.session_state or st.session_state.graph is None:
        answer = (
            "No documents loaded. Upload PDFs or add a URL using the sidebar "
            "and enter API keys."
        )
    elif not moderation(prompt):
        answer = "I'm sorry, I can't help with that request."
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                result = st.session_state.graph.run(
                    prompt,
                    source_files=st.session_state.get("source_files", []),
                    paper_metadata=st.session_state.get("paper_metadata", {}),
                    chat_history=st.session_state.messages[:-1],
                )
                answer = result.get("answer", "No answer generated.")
            st.markdown(answer)
            render_copy_button(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
