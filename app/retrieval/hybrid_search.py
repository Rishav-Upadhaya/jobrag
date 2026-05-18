from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import psycopg2

from app.core.config import settings
from app.infrastructure.db import repository
from app.infrastructure.embeddings.bge_m3 import get_embedder
from app.retrieval.rerank import get_jina_reranker

logger = logging.getLogger(__name__)


def deduplicate_by_job(
    chunks: list[dict],
    max_chunks_per_job: int = 2,
) -> list[dict]:
    seen: dict[str, int] = defaultdict(int)
    result = []
    for chunk in chunks:
        job_id = chunk.get("job_id")
        if seen[job_id] < max_chunks_per_job:
            seen[job_id] += 1
            result.append(chunk)
    return result


def rerank_chunks(
    chunks: list[dict],
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    if not chunks:
        return []

    if top_k is None:
        top_k = settings.TOP_K_RERANK
    if threshold is None:
        threshold = settings.JINA_RERANK_THRESHOLD

    try:
        documents = [chunk.get("chunk_text", "") for chunk in chunks]
        if not documents:
            return chunks[:top_k]

        reranker = get_jina_reranker()
        reranked_results = reranker.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            threshold=threshold,
        )

        if not reranked_results:
            logger.warning("Jina returned no results above threshold; using original order")
            return chunks[:top_k]

        reranked_chunks = []
        for result in reranked_results:
            index = result.get("index")
            if index is not None and index < len(chunks):
                chunk = chunks[index].copy()
                chunk["rerank_score"] = result.get("relevance_score", 0.0)
                reranked_chunks.append(chunk)

        return reranked_chunks

    except Exception as e:
        logger.error("Jina reranking failed: %s; falling back to original order", e)
        return chunks[:top_k]


async def hybrid_search(
    conn: psycopg2.extensions.connection,
    query_text: str,
    filters: dict,
    top_k_vector: int | None = None,
    final_top_k: int = 5,
) -> list[dict]:
    if top_k_vector is None:
        top_k_vector = settings.LLAMA_CLOUD_TOP_K

    try:
        job_id_whitelist = repository.get_matching_job_ids(conn, filters)

        if job_id_whitelist is not None and len(job_id_whitelist) == 0:
            job_id_whitelist = None

        query_emb = get_embedder().embed_query(query_text)
        
        vector_results = repository.vector_search(
            conn, 
            query_embedding=query_emb,
            job_id_whitelist=job_id_whitelist,
            top_k=top_k_vector
        )

        results = deduplicate_by_job(vector_results, max_chunks_per_job=2)

        candidate_count = min(
            len(results),
            max(final_top_k * 3, settings.TOP_K_RERANK, final_top_k),
        )
        candidates = results[:candidate_count]

        if settings.JINA_API_KEY and candidates:
            reranked = rerank_chunks(
                candidates,
                query_text,
                top_k=min(len(candidates), max(final_top_k, settings.TOP_K_RERANK)),
                threshold=settings.JINA_RERANK_THRESHOLD,
            )
            if reranked:
                reranked_ids = {c.get("chunk_id") for c in reranked}
                backfill = [c for c in candidates if c.get("chunk_id") not in reranked_ids]
                return (reranked + backfill)[:final_top_k]

        return candidates[:final_top_k]

    except Exception as e:
        logger.error("Error in hybrid_search: %s", e)
        raise