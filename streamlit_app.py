import os
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentHandler
from src.graph_builder.graph import GraphBuilder
from src.vector_store.store import VectorStore

load_dotenv()

st.set_page_config(page_title="NoteForge RAG", page_icon="🧠")
st.title("NoteForge — RAG Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []


def ingest_documents():
    data_dir = Path("data")
    if not data_dir.is_dir():
        st.info("Create a data/ folder with PDFs to get started.")
        return

    pdfs = sorted(data_dir.glob("*.pdf")) + sorted(data_dir.glob("*.txt"))
    if not pdfs:
        st.info("Drop PDFs into the data/ folder and restart.")
        return

    handler = DocumentHandler()
    all_docs = []

    with st.status("📄 Loading documents…", expanded=True) as status:
        for pdf in pdfs:
            status.write(f"Loading {pdf.name} …")
            loaded = (
                handler.pdf_loader(str(pdf))
                if pdf.suffix == ".pdf"
                else handler.text_loader(str(pdf))
            )
            all_docs.extend(loaded)

        status.write("Splitting into chunks …")
        chunks = handler.doc_splitter(all_docs)

        source_counts = Counter(doc.metadata.get("source", "unknown") for doc in chunks)

        status.write(f"Embedding {len(chunks)} chunks …")
        vs = VectorStore()
        vs.create_retriever(chunks)
        st.session_state.retriever = vs.get_retriever()

        # Store ground-truth file list — passed into every graph run so the
        # agent always knows exactly what was uploaded, regardless of what
        # retrieval returns for a given query.
        st.session_state.source_files = [pdf.name for pdf in pdfs]

        status.write("Extracting document summaries …")
        llm = Config.get_llm()
        st.session_state.doc_summaries = handler.extract_summaries(all_docs, llm)

        status.write("Building graph …")
        builder = GraphBuilder(st.session_state.retriever, llm)
        builder.build()
        st.session_state.graph = builder
        st.session_state.chunk_counts = source_counts

        status.update(
            label=f"✅ Ready — {len(pdfs)} docs, {len(chunks)} chunks",
            state="complete",
        )


with st.sidebar:
    st.header("🔑 API Keys")
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

    if api_key and "graph" not in st.session_state:
        ingest_documents()

    if "chunk_counts" in st.session_state:
        st.caption("📊 Index")
        for source, count in st.session_state.chunk_counts.most_common():
            st.caption(f"  {Path(source).name} — {count} chunks")

    st.divider()
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "graph" not in st.session_state or st.session_state.graph is None:
        answer = (
            "⚠️ No documents loaded. Add PDFs to the data/ folder and enter API keys."
        )
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                result = st.session_state.graph.run(
                    prompt,
                    source_files=st.session_state.get("source_files", []),
                    doc_summaries=st.session_state.get("doc_summaries", {}),
                    chat_history=st.session_state.messages[:-1],
                )
                answer = result.get("answer", "No answer generated.")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
