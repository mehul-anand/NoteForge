from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """For the vector store :)"""

    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_retreiver(self, documents: List[Document]):
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError(
                "Vector store is not initialised, use the create_vectorstore first"
            )
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if self.retriever is None:
            raise ValueError(
                "Vector store is not initialised, use the create_vectorstore first"
            )
        return self.retriever.invoke(query)
