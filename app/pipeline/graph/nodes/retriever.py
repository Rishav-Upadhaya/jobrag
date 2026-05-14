"""Retriever node.

Performs hybrid search combining vector and keyword search to retrieve relevant job chunks.

Single Responsibility: Retrieve relevant jobs based on query and parameters.
"""

import logging

from app.db.connection import get_conn
from app.services.search_service import hybrid_search
from app.config import settings
from app.pipeline.graph.state.graph_state import GraphState
from app.pipeline.ingestion.embedder import get_embedder

logger = logging.getLogger(__name__)


def retriever(state: GraphState) -> dict[str, object]:
    """
    Retrieve relevant job chunks using hybrid search.

    Expects reasoning_output from reasoning_agent node containing:
    - retriever_query: The query to search for
    - top_k: Number of results to retrieve
    - filters: Metadata filters (job_level, job_category, job_location)

    The node is intentionally pure search logic: it embeds the query,
    performs hybrid retrieval, and returns only the state keys it modifies.
    """
    # Get retrieval parameters from reasoning_agent output
    reasoning_output = state.get("reasoning_output")
    if reasoning_output:
        query = reasoning_output.retriever_query
        top_k = reasoning_output.top_k
        filters = reasoning_output.filters or {}
    else:
        # Fallback for compatibility
        query = state.get("resolved_query") or state.get("user_query") or state.get("query", "")
        top_k = state.get("top_k", 5)
        filters = state.get("filters", {})

    retry_count = state.get("retry_count", 0)
    top_k = max(1, min(int(top_k), 50))  # Clamp to 1-50

    if not query:
        logger.warning("Retriever: No query provided")
        return {
            "retrieved_chunks": [],
            "retry_count": retry_count,
        }

    logger.info(
        f"Retrieving chunks: query_len={len(query)}, top_k={top_k}, "
        f"filters={filters}, retry_count={retry_count}"
    )

    try:
        # Embed the query
        embedder = get_embedder()
        query_embedding = embedder.embed_batch([query])[0]

        if not query_embedding:
            logger.error("Embedder returned empty query embedding")
            return {
                "retrieved_chunks": [],
                "retry_count": retry_count,
            }

        logger.debug(f"Query embedding dimension: {len(query_embedding)}")

        # Perform hybrid search
        with get_conn() as conn:
            retrieved_chunks = hybrid_search(
                conn=conn,
                query_text=query,
                query_embedding=query_embedding,
                filters=filters,
                top_k_vector=max(settings.TOP_K_VECTOR, top_k * 4),
                top_k_keyword=max(settings.TOP_K_KEYWORD, top_k * 4),
                final_top_k=top_k,
            )

            # Fallback: retry without filters if no results found
            if not retrieved_chunks and filters:
                logger.info(f"No chunks found with filters={filters}; retrying without filters")
                retrieved_chunks = hybrid_search(
                    conn=conn,
                    query_text=query,
                    query_embedding=query_embedding,
                    filters={},
                    top_k_vector=max(settings.TOP_K_VECTOR, top_k * 4),
                    top_k_keyword=max(settings.TOP_K_KEYWORD, top_k * 4),
                    final_top_k=top_k,
                )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        return {
            "retrieved_chunks": retrieved_chunks,
            "retry_count": retry_count,
        }

    except Exception as exc:
        logger.exception(f"Error in retriever: {exc}")
        raise
