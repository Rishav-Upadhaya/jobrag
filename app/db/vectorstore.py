"""vectorstore.py — Deprecated shim.

After the LlamaCloud refactor, this file's responsibilities have been split:

  MOVED TO → app/db/repository.py:
    bulk_upsert_jobs(), upsert_job()
    bulk_upsert_chunks(), upsert_chunk()
    keyword_search()
    get_matching_job_ids()    ← new

  MOVED TO → app/pipeline/ingestion/llama_indexer.py:
    llama_vector_search()     ← replaces vector_search()
    index_jobs()              ← replaces bulk_upsert_chunks (embedding side)

  MOVED TO → app/services/search_service.py:
    hybrid_search()
    rerank_chunks()
    deduplicate_by_job()

This file re-exports the repository functions under their old names so that
any remaining direct imports of vectorstore.* continue to work during the
transition. Once all callers are updated, this file can be deleted.

DO NOT add new logic here.
"""

from __future__ import annotations

import logging

# Re-export repository functions under old names for backward compatibility
from app.db.repository import (
    bulk_upsert_jobs,
    upsert_job,
    bulk_upsert_chunks,
    upsert_chunk,
    keyword_search,
    get_matching_job_ids,
)

# Re-export service functions
from app.services.search_service import (
    hybrid_search,
    rerank_chunks,
    deduplicate_by_job,
)

# Re-export LlamaCloud vector search under the old name
from app.pipeline.ingestion.llama_indexer import llama_vector_search as vector_search

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
