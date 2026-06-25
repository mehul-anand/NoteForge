from typing import List

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
