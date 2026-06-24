from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config.config import Config


class VectorStore:
    """For the vector store"""

    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self, documents: List[Document]):
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        # k=10 ensures chunks from all uploaded papers come back
        # (old k=4 could miss entire papers when multiple docs are indexed)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": Config.RETRIEVER_K}
        )

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError(
                "Vector store is not initialised — call create_retriever() first."
            )
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        if self.retriever is None:
            raise ValueError(
                "Vector store is not initialised — call create_retriever() first."
            )
        return self.retriever.invoke(query)
