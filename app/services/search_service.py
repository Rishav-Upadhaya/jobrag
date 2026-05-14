"""Search service — orchestrates hybrid retrieval, fusion, and reranking.

This module contains the business logic for combining vector and keyword
search results into a final ranked list. It depends on the repository layer
(not raw DB calls) and pure utility functions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import psycopg2

from app.config import settings
from app.db import repository
from app.utils.jina_reranker import get_jina_reranker
from app.utils.rrf import rrf_fusion

logger = logging.getLogger(__name__)


def deduplicate_by_job(
    chunks: list[dict],
    max_chunks_per_job: int = 2,
) -> list[dict]:
    """Deduplicate chunks by job_id, keeping at most max_chunks_per_job per
    job. Preserves the order of fused_score (highest scoring chunks first).

    Args:
        chunks: List of chunk dicts with job_id field
        max_chunks_per_job: Maximum number of chunks to keep per job_id

    Returns:
        Filtered list of chunks preserving the input order
    """
    seen_jobs: dict[str, int] = defaultdict(int)
    result = []
    for chunk in chunks:
        job_id = chunk.get("job_id")
        if seen_jobs[job_id] < max_chunks_per_job:
            seen_jobs[job_id] += 1
            result.append(chunk)
    return result


def rerank_chunks(
    chunks: list[dict],
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Rerank chunks using Jina Reranking API.

    Args:
        chunks: List of chunk dicts with chunk_text field
        query: The search query
        top_k: Number of top results to return
                (defaults to settings.TOP_K_RERANK)
        threshold: Minimum relevance score threshold
                    (defaults to settings.JINA_RERANK_THRESHOLD)

    Returns:
        List of reranked chunks sorted by Jina relevance score
    """
    if not chunks:
        return []

    if top_k is None:
        top_k = settings.TOP_K_RERANK

    if threshold is None:
        threshold = settings.JINA_RERANK_THRESHOLD

    try:
        documents = [chunk.get("chunk_text", "") for chunk in chunks]

        if not documents:
            logger.warning("No documents to rerank")
            return chunks[:top_k]

        reranker = get_jina_reranker()
        reranked_results = reranker.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            threshold=threshold,
        )

        if not reranked_results:
            logger.warning(
                "No results from Jina reranker above threshold %s, "
                "returning original chunks",
                threshold,
            )
            return chunks[:top_k]

        reranked_chunks = []

        for result in reranked_results:
            index = result.get("index")
            if index is not None and index < len(chunks):
                chunk = chunks[index].copy()
                chunk["rerank_score"] = result.get("relevance_score", 0.0)
                reranked_chunks.append(chunk)

        logger.info(
            "Reranked %s chunks, returning %s results",
            len(chunks),
            len(reranked_chunks),
        )
        return reranked_chunks

    except Exception as e:
        logger.error(
            "Error in rerank_chunks: %s, falling back to original ranking", e
        )
        return chunks[:top_k]


def hybrid_search(
    conn: psycopg2.extensions.connection,
    query_text: str,
    query_embedding: list[float],
    filters: dict,
    top_k_vector: int = 20,
    top_k_keyword: int = 20,
    final_top_k: int = 5,
) -> list[dict]:
    """Hybrid search combining vector and keyword search with RRF fusion.

    This is the main retrieval entry point used by the retriever graph node.
    It orchestrates:
        1. Vector search (cosine similarity)
        2. Keyword search (full-text tsvector)
        3. RRF fusion
        4. Job-level deduplication
        5. Optional Jina reranking
        6. Final top-k selection

    Args:
        conn: Database connection with pgvector registered
        query_text: Query text for full-text search
        query_embedding: Query vector embedding
        filters: Dict with optional keys: job_level, job_category, job_location
        top_k_vector: Number of vector search results to consider
        top_k_keyword: Number of keyword search results to consider
        final_top_k: Number of final results to return

    Returns:
        List of dicts with keys: chunk_id, job_id, chunk_text, score,
        job_title, company_name, job_level, job_location, job_category
    """
    try:
        vector_results = repository.vector_search(
            conn, query_embedding, filters, top_k_vector
        )

        keyword_results = repository.keyword_search(
            conn, query_text, filters, top_k_keyword
        )

        fused = rrf_fusion([vector_results, keyword_results], k=60)

        fused = deduplicate_by_job(fused, max_chunks_per_job=2)

        candidate_count = min(
            len(fused),
            max(final_top_k * 3, settings.TOP_K_RERANK, final_top_k),
        )
        candidates = fused[:candidate_count]

        if settings.JINA_API_KEY and candidates:
            reranked = rerank_chunks(
                candidates,
                query_text,
                top_k=min(
                    len(candidates),
                    max(final_top_k, settings.TOP_K_RERANK),
                ),
                threshold=settings.JINA_RERANK_THRESHOLD,
            )
            if reranked:
                reranked_ids = {chunk.get("chunk_id") for chunk in reranked}
                backfill = [
                    chunk
                    for chunk in candidates
                    if chunk.get("chunk_id") not in reranked_ids
                ]
                return (reranked + backfill)[:final_top_k]

        return candidates[:final_top_k]
    except Exception as e:
        logger.error("Error in hybrid_search: %s", e)
        raise