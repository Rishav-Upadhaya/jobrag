"""
Application configuration via Pydantic Settings.

Clean, minimal config after the LlamaCloud + Jina refactor.
Dead keys removed: EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
RERANKER_MODEL, COHERE_API_KEY, HF_* settings.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── LlamaCloud (vector index + embeddings) ────────────────────────────────
    LLAMA_CLOUD_API_KEY: str
    LLAMA_CLOUD_INDEX_NAME: str = "leapfrog-jobs"
    LLAMA_CLOUD_PROJECT_NAME: str = "Default"
    # How many candidates to pull from LlamaCloud before Jina reranking
    LLAMA_CLOUD_TOP_K: int = 40

    # ── LLM providers ────────────────────────────────────────────────────────
    LLM_PROVIDER: str                          # "google" | "openai" | "openrouter"
    LLM_SYNTHESIZER_MODEL: str
    LLM_CLASSIFIER_MODEL: str
    LLM_JUDGE_MODEL: str

    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: Optional[str] = None

    # ── Jina reranker ─────────────────────────────────────────────────────────
    JINA_API_KEY: str                          # Required — no reranker without this
    JINA_RERANKER_MODEL: str = "jina-reranker-v2-base-multilingual"
    JINA_RERANK_THRESHOLD: float = 0.3        # Lowered from 0.5 to avoid empty results

    # ── Retrieval knobs ───────────────────────────────────────────────────────
    # Keyword search (pgvector FTS — still used alongside LlamaCloud vector)
    TOP_K_KEYWORD: int = 40
    TOP_K_RERANK: int = 10                    # Final candidates after Jina rerank
    RRF_K: int = 60                           # RRF smoothing constant

    # ── Judge ─────────────────────────────────────────────────────────────────
    JUDGE_PASS_THRESHOLD: float = 0.7
    JUDGE_MAX_RETRIES: int = 2

    # ── LLM generation ────────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = 0.1


settings = Settings()
logging.getLogger(__name__).info(
    "Config loaded | LLM provider=%s | LlamaCloud index=%s",
    settings.LLM_PROVIDER,
    settings.LLAMA_CLOUD_INDEX_NAME,
)