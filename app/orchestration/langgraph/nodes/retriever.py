import logging

from app.infrastructure.db.connection import get_conn
from app.retrieval.hybrid_search import hybrid_search
from app.core.config import settings
from app.orchestration.langgraph.state.graph_state import GraphState

logger = logging.getLogger(__name__)


async def retriever(state: GraphState) -> dict[str, object]:
    """Retrieve relevant job chunks using hybrid search.

    Expects reasoning_output from reasoning_agent node containing:
    - retriever_query: The query to search for
    - top_k:          Number of results to retrieve
    - filters:        Metadata filters (7 fields: job_level, job_category,
                      job_location, company_name, job_title, date_order, date_after)

    Returns only the state keys this node modifies.
    """
    # ── Unpack reasoning_output ───────────────────────────────────────────────
    reasoning_output = state.get("reasoning_output")
    if reasoning_output:
        query   = reasoning_output.retriever_query
        top_k   = reasoning_output.top_k
        filters = reasoning_output.filters or {}
    else:
        # Fallback for direct invocation / tests
        query   = state.get("resolved_query") or state.get("user_query") or state.get("query", "")
        top_k   = state.get("top_k", 5)
        filters = state.get("filters", {})

    retry_count = state.get("retry_count", 0)
    top_k = max(1, min(int(top_k), 50))  # clamp to 1-50

    if not query:
        logger.warning("Retriever: No query provided")
        return {"retrieved_chunks": [], "retry_count": retry_count}

    logger.info(
        "Retriever: query_len=%d, top_k=%d, filters=%s, retry=%d",
        len(query), top_k, filters, retry_count,
    )

    try:
        with get_conn() as conn:
            retrieved_chunks = await hybrid_search(
                conn=conn,
                query_text=query,
                filters=filters,
                top_k_vector=max(settings.LLAMA_CLOUD_TOP_K, top_k * 4),
                final_top_k=top_k,
            )

            # ── Graceful fallback: retry without filters if nothing returned ──
            if not retrieved_chunks and filters:
                logger.info(
                    "No chunks with filters=%s; retrying without filters", filters
                )
                retrieved_chunks = await hybrid_search(
                    conn=conn,
                    query_text=query,
                    filters={},
                    top_k_vector=max(settings.LLAMA_CLOUD_TOP_K, top_k * 4),
                    final_top_k=top_k,
                )

        logger.info("Retriever: returned %d chunks", len(retrieved_chunks))
        return {
            "retrieved_chunks": retrieved_chunks,
            "retry_count": retry_count,
        }

    except Exception as exc:
        logger.exception("Error in retriever: %s", exc)
        raise
