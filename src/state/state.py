from typing import Any, Dict, List, Optional

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
    # Structured per-document metadata (PaperMetadata) — injected into agent
    # context so exact fields (authors[], title, year, venue, methods) are
    # always available regardless of retrieval quality.
    paper_metadata: Dict[str, Any] = {}
    chat_history: List[Dict[str, str]] = []
    sub_queries: List[str] = []
    rewritten_queries: List[str] = []
    # Intent routing: one of qa / compare / review / gaps (default qa).
    task_type: str = "qa"
    # Structured synthesis output (ComparisonMatrix/RelatedWork/ResearchGaps)
    # as a populated dict — kept for evals and future UI, the rendered markdown
    # lives in `answer`.
    artifact: Optional[Dict[str, Any]] = None
