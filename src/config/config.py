import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# importing environment variables
load_dotenv()


class Config:

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Model config
    LLM_MODEL = "openai:gpt-4o-mini"

    # Document processing
    # Increased from 500/50 → larger chunks capture full paper metadata
    # (title, authors, institution) in one chunk instead of fragmenting it
    CHUNK_SIZE = 2500
    CHUNK_OVERLAP = 150

    # Retriever config
    # Increased from 4 → ensures chunks from ALL uploaded papers are returned
    RETRIEVER_K = 10

    @classmethod
    def get_llm(cls):
        """Initialise and return the LLM Model"""
        return init_chat_model(cls.LLM_MODEL)
