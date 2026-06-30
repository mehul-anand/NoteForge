from typing import Dict, List

from langchain_core.documents import Document
from pydantic import BaseModel


class State(BaseModel):
    """State Object for the RAG Workflow"""

    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    # Ground truth list of uploaded filenames — passed from streamlit into every
    # graph run so the agent always knows what files exist, regardless of what
    # the retriever happens to return for a given query.
    source_files: List[str] = []
    # Per-document LLM summaries (1 sentence each) — injected into agent context
    # so metadata (authors, titles, doc type) is always available regardless of
    # retrieval quality.
    doc_summaries: Dict[str, str] = {}
    chat_history: List[Dict[str, str]] = []
    sub_queries: List[str] = []
