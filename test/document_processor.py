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
