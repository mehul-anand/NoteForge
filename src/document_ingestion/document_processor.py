"""Document processing module for loading and splitting documents"""

from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (PyPDFDirectoryLoader,
                                                  PyPDFLoader, TextLoader,
                                                  WebBaseLoader)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentHandler:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def url_loader(self, url: str) -> List[Document]:
        """Loading documents for chunking from URLs"""
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
        loader = PyPDFLoader(str(file_path))  # or pass the data folder directly
        return loader.load()

    def all_document_loader(self, sources: List[str]) -> List[Document]:
        """Loads everything we got as sources"""
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.url_loader(src))
            path = Path("data")
            if path.is_dir():
                docs.extend(self.pdf_dir_loader(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.text_loader(path))
            else:
                raise ValueError(
                    f"Unsupported source stype: {src} \n Please use URL, text files or PDF directory"
                )
        return docs
