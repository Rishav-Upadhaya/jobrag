
from __future__ import annotations

import logging

# Re-export repository functions under old names for backward compatibility
from app.infrastructure.db.repository import (
    bulk_upsert_jobs,
    upsert_job,
    bulk_upsert_chunks,
    upsert_chunk,
    keyword_search,
    get_matching_job_ids,
)

# Re-export service functions
from app.retrieval.hybrid_search import (
    hybrid_search,
    rerank_chunks,
    deduplicate_by_job,
)

# Re-export LlamaCloud vector search under the old name
from app.ingestion.llama_indexer import llama_vector_search as vector_search

logger = logging.getLogger(__name__)
logger.warning(
    "app.db.vectorstore is deprecated. Import directly from "
    "app.db.repository, app.services.search_service, or "
    "app.pipeline.ingestion.llama_indexer instead."
)

__all__ = [
    "bulk_upsert_jobs",
    "upsert_job",
    "bulk_upsert_chunks",
    "upsert_chunk",
    "keyword_search",
    "get_matching_job_ids",
    "hybrid_search",
    "rerank_chunks",
    "deduplicate_by_job",
    "vector_search",
]
