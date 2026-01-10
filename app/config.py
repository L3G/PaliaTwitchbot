"""Configuration settings for the Palia Wiki Q&A Bot."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # ChromaDB Configuration
    chroma_persist_directory: str = str(PROJECT_ROOT / "data" / "chroma_db")
    chroma_collection_name: str = "palia_wiki"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Rate Limiting
    rate_limit_requests: int = 30
    rate_limit_window: int = 60  # seconds

    # Caching
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000

    # RAG Settings
    retrieval_k: int = 5  # Number of chunks to retrieve
    max_response_length: int = 400  # Nightbot character limit

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
