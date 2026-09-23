"""Document processing module for loading and splitting documents"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.config.config import Config


def normalize_source_key(source: str) -> str:
    """Canonical key for a document source.

    PDFs/TXT keys are their basename; URL sources are the full URL so that
    `source_files`, `paper_metadata`, and retriever doc metadata all agree
    (a partial path like `2307.03172v3` would never match the URL)."""
    if source.startswith(("http://", "https://")):
        return source
    return Path(str(source)).name


def relabel_sources(documents, source: str) -> None:
    """Set every document's source metadata to a canonical label, in place.

    Uploaded files are loaded from temporary paths (e.g.
    `/tmp/tmpj_fmf38p.pdf`); relabelling keeps chunk counts, paper metadata
    keys, and the agent's file matching all keyed by the original filename
    the user actually uploaded."""
    for doc in documents:
        doc.metadata["source"] = source


class PaperMetadata(BaseModel):
    """Structured metadata extracted from the first page of a document."""

    title: str = ""
    authors: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    methods: List[str] = Field(default_factory=list)
    contributions: List[str] = Field(default_factory=list)
    key_results: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    source: str = ""


class DocumentHandler:
    """Handles document loading and processing"""

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def url_loader(self, url: str) -> List[Document]:
        """Load documents for chunking from URLs"""
        loader = WebBaseLoader(url)
        return loader.load()

    def pdf_dir_loader(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def text_loader(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def pdf_loader(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyMuPDFLoader(str(file_path))
        return loader.load()

    def doc_splitter(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    def extract_summaries(self, documents: List[Document], llm) -> dict:
        grouped = {}
        for doc in documents:
            src = normalize_source_key(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", 0)
            if src not in grouped or page < grouped[src]["page"]:
                grouped[src] = {"page": page, "content": doc.page_content[:2000]}

        summaries = {}
        for filename, info in grouped.items():
            content = info["content"]
            prompt = (
                "Summarize what this document is about in 1 sentence. "
                "Focus on the topic, document type, and key entities "
                "(people, organizations, dates). "
                "Start directly with the summary — no preamble.\n\n"
                f"Document content:\n{content}"
            )
            try:
                resp = llm.invoke(prompt)
                summaries[filename] = resp.content.strip()
            except Exception:
                summaries[filename] = ""
        return summaries

    @staticmethod
    def _fill_year(paper: "PaperMetadata", filename: str) -> "PaperMetadata":
        """Fallback: pull the year from the filename (e.g. CBF_UAV_2024.pdf)."""
        if paper.year is None:
            match = re.search(r"(19[7-9]\d|20[0-2]\d)", filename)
            if match:
                paper.year = int(match.group(1))
        return paper

    def extract_metadata(self, documents: List[Document], llm) -> Dict[str, PaperMetadata]:
        """Extract structured PaperMetadata per source file.

        Groups loaded documents by source file, takes the first page (lowest
        page number), and asks the LLM for structured output. Retries once
        before degrading to the old 1-line summary, so the pipeline never
        returns nothing.
        """
        grouped = {}
        for doc in documents:
            src = normalize_source_key(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", 0)
            if src not in grouped or page < grouped[src]["page"]:
                grouped[src] = {"page": page, "content": doc.page_content[:3000]}

        try:
            structured_llm = llm.with_structured_output(PaperMetadata)
        except Exception:
            structured_llm = None

        metadata = {}
        for filename, info in grouped.items():
            paper = self._extract_metadata_for_file(
                filename, info["content"], llm, structured_llm
            )
            self._fill_year(paper, filename)
            metadata[filename] = paper
        return metadata

    def _extract_metadata_for_file(
        self, filename: str, content: str, llm, structured_llm
    ) -> PaperMetadata:
        """One structured attempt, then a retry, then degraded summary fallback."""
        if structured_llm is not None:
            prompt = (
                "Extract paper metadata from this first page of a "
                "document. Fill every field you can determine. "
                "affiliations = institutions/organizations the authors "
                "belong to. contributions = what the work proposes/does. "
                "key_results = concrete findings or validation outcomes. "
                "Use empty strings/lists and null year where the info is "
                f"absent or unclear. Do not invent facts.\n\nDocument content:\n{content}"
            )
            for _ in range(2):  # one retry on failure
                try:
                    resp = structured_llm.invoke(prompt)
                    data = resp.model_dump()
                    data["source"] = filename
                    return PaperMetadata(**data)
                except Exception:
                    continue
        return self._metadata_from_summary(filename, content, llm)

    @staticmethod
    def _metadata_from_summary(filename: str, content: str, llm) -> PaperMetadata:
        """Degraded fallback: wrap the old 1-line summary into a PaperMetadata."""
        try:
            prompt = (
                "Summarize what this document is about in 1 sentence. "
                "Focus on the topic, document type, and key entities "
                "(people, organizations, dates). "
                "Start directly with the summary — no preamble.\n\n"
                f"Document content:\n{content}"
            )
            resp = llm.invoke(prompt)
            summary = resp.content.strip()
        except Exception:
            summary = ""
        return PaperMetadata(source=filename, title=filename, contributions=[summary])
