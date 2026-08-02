"""Application settings loaded from environment / .env file."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BROKERIQ_",
        extra="ignore",
        case_sensitive=False,
    )

    # Runtime
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # LLM
    model: str = "openrouter/google/gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    openrouter_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "BROKERIQ_OPENROUTER_API_KEY")
    )
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "BROKERIQ_GEMINI_API_KEY")
    )
    groq_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "BROKERIQ_GROQ_API_KEY")
    )

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("LANGSMITH_API_KEY", "BROKERIQ_LANGSMITH_API_KEY")
    )
    langsmith_project: str = "brokeriq"

    # Local infra
    postgres_dsn: str = Field(
        default="postgresql://brokeriq:brokeriq@localhost:5432/brokeriq",
        validation_alias=AliasChoices("POSTGRES_DSN", "BROKERIQ_POSTGRES_DSN"),
    )
    qdrant_url: str = Field(
        default="http://localhost:6333", validation_alias=AliasChoices("QDRANT_URL", "BROKERIQ_QDRANT_URL")
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", validation_alias=AliasChoices("REDIS_URL", "BROKERIQ_REDIS_URL")
    )

    # Application
    corpus_dir: str = "data/corpus"
    cache_ttl: int = 3600
    web_search_provider: Literal["ddgs", "tavily"] = "ddgs"
    tavily_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("TAVILY_API_KEY", "BROKERIQ_TAVILY_API_KEY")
    )

    def export_llm_env(self) -> None:
        """Make provider keys visible to litellm, which reads standard env vars."""
        if self.openrouter_api_key:
            os.environ.setdefault("OPENROUTER_API_KEY", self.openrouter_api_key)
        if self.gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", self.gemini_api_key)
        if self.groq_api_key:
            os.environ.setdefault("GROQ_API_KEY", self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.export_llm_env()
    return settings
