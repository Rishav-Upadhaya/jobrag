from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
        model_config = SettingsConfigDict(
                env_file=".env",
                env_file_encoding="utf-8",
                extra="ignore",
        )

        DATABASE_URL: str

        EMBEDDING_PROVIDER: str
        EMBEDDING_MODEL: str
        EMBEDDING_DIMENSION: int

        LLM_PROVIDER: str
        LLM_SYNTHESIZER_MODEL: str
        LLM_CLASSIFIER_MODEL: str
        LLM_JUDGE_MODEL: str

        GOOGLE_API_KEY: Optional[str] = None
        OPENAI_API_KEY: Optional[str] = None
        OPENROUTER_API_KEY: Optional[str] = None
        OPENROUTER_BASE_URL: Optional[str] = None

        JINA_API_KEY: Optional[str] = None
        JINA_RERANKER_MODEL: str = "jina-reranker-v2-base-multilingual"

        TOP_K_VECTOR: int
        TOP_K_KEYWORD: int
        TOP_K_RERANK: int
        RERANKER_MODEL: str
        RRF_K: int
        JINA_RERANK_THRESHOLD: float = 0.5

        JUDGE_PASS_THRESHOLD: float
        JUDGE_MAX_RETRIES: int
        EMBEDDING_BATCH_SIZE: int = 256
        
        # Low temperature for faster, more deterministic JSON output
        LLM_TEMPERATURE: float = 0.1


settings = Settings()
import logging
logging.getLogger(__name__).info(f"Loaded LLM_CLASSIFIER_MODEL: {settings.LLM_CLASSIFIER_MODEL}")
