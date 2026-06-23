import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.document_ingestion.document_processor import DocumentHandler
from src.graph_builder.graph import GraphBuilder
from src.vector_store.store import VectorStore

load_dotenv()

st.set_page_config(page_title="NoteForge RAG", page_icon="🧠")
st.title("NoteForge — RAG Q&A")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    st.header("📄 Documents")

    urls = st.text_area("URLs (one per line)", placeholder="https://example.com/doc")
    uploaded = st.file_uploader(
        "Upload PDFs / TXT", accept_multiple_files=True, type=["pdf", "txt"]
    )

    if st.button("🚀 Ingest & Build Graph", type="primary"):
        if not api_key:
            st.error("Enter an API key first")
            st.stop()

        handler = DocumentHandler()
        all_docs = []

        with st.status("Processing…", expanded=True) as status:
            if urls:
                for url in filter(None, urls.strip().split("\n")):
                    status.write(f"Loading {url} …")
                    docs = handler.url_loader(url)
                    all_docs.extend(handler.doc_splitter(docs))

            if uploaded:
                for f in uploaded:
                    suffix = Path(f.name).suffix
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False
                    ) as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name
                    status.write(f"Loading {f.name} …")
                    loaded = (
                        handler.pdf_loader(tmp_path)
                        if suffix == ".pdf"
                        else handler.text_loader(tmp_path)
                    )
                    all_docs.extend(handler.doc_splitter(loaded))
                    os.unlink(tmp_path)

            if not all_docs:
                st.warning("No documents loaded")
                st.stop()

            status.write(f"Embedding {len(all_docs)} chunks …")
            vs = VectorStore()
            vs.create_retreiver(all_docs)
            st.session_state.retriever = vs.get_retriever()

            status.write("Building graph …")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            builder = GraphBuilder(st.session_state.retriever, llm)
            builder.build()
            st.session_state.graph = builder

            status.update(label="✅ Ready — ask away!", state="complete")

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

    if st.session_state.graph is None:
        answer = "⚠️ Ingest documents first via the sidebar."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                result = st.session_state.graph.run(prompt)
                answer = result.get("answer", "No answer generated.")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
