"""Document processing module for loading and splitting documents"""

from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.config import Config


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
            src = Path(doc.metadata.get("source", "")).name
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
