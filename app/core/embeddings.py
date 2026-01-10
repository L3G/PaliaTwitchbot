"""OpenAI embeddings wrapper."""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance."""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
