import os
from functools import cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from openai import OpenAI

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
    CHUNK_OVERLAP = 200

    # Retriever config
    # Increased from 4 → ensures chunks from ALL uploaded papers are returned
    RETRIEVER_K = 15

    # Moderation flag
    MODERATION_FLAGGED = "FLAGGED_REASON"

    @classmethod
    @cache
    def get_llm(cls):
        """Initialise and return the LLM Model"""
        return init_chat_model(cls.LLM_MODEL)

    @classmethod
    @cache
    def get_moderator(cls):
        """Initialise the OpenAI moderation client (cached)."""
        return OpenAI(api_key=cls.OPENAI_API_KEY)


@cache
def moderation(text: str) -> bool:
    """Run OpenAI moderation on input. Returns True if input is SAFE.

    Flagged (unsafe) input -> False so the caller can refuse to serve it.
    Never raises: on moderation API errors we allow the request through
    (fail-open; the downstream agent prompt is already hardened).
    """
    try:
        client = Config.get_moderator()
        result = client.moderations.create(model="omni-moderation-latest", input=text)
        return not bool(result.results[0].flagged)
    except Exception:
        return True
