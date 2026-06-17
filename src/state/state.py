from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel


class State(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str
