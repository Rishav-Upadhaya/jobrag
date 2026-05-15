"""Reasoning agent node.

Understands the user query with conversation context, extracts retrieval
parameters (top_k, filters), and generates a standalone retriever query.

Filters extracted (7 fields):
  - job_level, job_category, job_location  (original 3)
  - company_name, job_title               (new — SQL LIKE filters)
  - date_order                             (new — "desc"/"asc" → ORDER BY)
  - date_after                             (new — ISO date → WHERE pub_date >)

Single Responsibility: Query understanding and parameter extraction.
"""

import logging
from typing import Any

from langchain.messages import SystemMessage
from app.llm.llm_client import get_llm
from app.pipeline.graph.state.graph_state import GraphState
from app.pipeline.graph.schemas.reasoning import ReasoningOutput
from app.pipeline.graph.utils.messages import build_history, extract_user_query
from app.pipeline.graph.utils.json import parse_json_object
from app.pipeline.graph.prompts.reasoning_agent import REASONING_AGENT_PROMPT

logger = logging.getLogger(__name__)

# All valid filter keys the LLM may return — anything else is stripped.
_VALID_FILTER_KEYS = {
    "job_level",
    "job_category",
    "job_location",
    "company_name",
    "job_title",
    "date_order",
    "date_after",
}


def _sanitize_filters(raw: Any) -> dict:
    """Strip unknown keys and null/empty values from LLM filter output."""
    if not isinstance(raw, dict):
        return {}
    return {
        k: v
        for k, v in raw.items()
        if k in _VALID_FILTER_KEYS and v and str(v).strip()
    }


def reasoning_agent(state: GraphState) -> dict[str, Any]:
    """Analyze user query and conversation context to extract retrieval parameters.

    Returns only the keys this node modifies.

    Responsibilities:
    - Extract top_k from query (default 5, range 1-50)
    - Extract filters: job_level, job_category, job_location,
                       company_name, job_title, date_order, date_after
    - Generate standalone retriever_query
    - Provide reasoning explanation

    Args:
        state: GraphState containing messages, query, and conversation context

    Returns:
        dict with keys: reasoning_output, retry_count, resolved_query
    """
    messages = state.get("messages", [])
    user_query = extract_user_query(messages) if messages else state.get("query", "")
    retry_count = state.get("retry_count", 0)

    if not user_query:
        logger.warning("Reasoning agent: No user query provided")
        return {
            "reasoning_output": ReasoningOutput(
                top_k=5,
                filters={},
                retriever_query="",
                reasoning="No query provided",
            ),
            "retry_count": retry_count,
        }

    logger.info("Reasoning agent: Analyzing query: %s...", user_query[:80])

    try:
        # Build conversation history for context (limit to 4 messages)
        history = (
            build_history(messages, limit=4)
            if messages
            else "No previous conversation."
        )

        # Format prompt with user query and history
        formatted_prompt = REASONING_AGENT_PROMPT.format(
            query=user_query,
            history=history,
        )

        # Use classifier-tier LLM (fast + cheap); 768 tokens is enough for 7 fields
        llm = get_llm("classifier", max_tokens=768)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        # Parse structured response
        structured_response = parse_json_object(content)
        logger.debug("Reasoning agent response: %s", structured_response)

        # ── top_k ─────────────────────────────────────────────────────────────
        top_k = structured_response.get("top_k", 5)
        top_k = max(1, min(int(top_k), 50))

        # ── filters ───────────────────────────────────────────────────────────
        raw_filters = structured_response.get("filters", {})
        filters = _sanitize_filters(raw_filters)

        # ── retriever_query ───────────────────────────────────────────────────
        retriever_query = str(
            structured_response.get("retriever_query", user_query)
        ).strip()
        if not retriever_query:
            retriever_query = user_query

        # ── reasoning ─────────────────────────────────────────────────────────
        reasoning = str(structured_response.get("reasoning", "")).strip()

        # Build output
        reasoning_output = ReasoningOutput(
            top_k=top_k,
            filters=filters,
            retriever_query=retriever_query,
            reasoning=reasoning,
        )

        logger.info(
            "Reasoning agent complete: top_k=%d, filters=%s, query_len=%d",
            top_k,
            filters,
            len(retriever_query),
        )

        return {
            "reasoning_output": reasoning_output,
            "retry_count": retry_count + 1,
            "resolved_query": retriever_query,
        }

    except Exception as exc:
        logger.exception("Error in reasoning_agent: %s", exc)
        return {
            "reasoning_output": ReasoningOutput(
                top_k=5,
                filters={},
                retriever_query=user_query,
                reasoning="Error during reasoning",
            ),
            "retry_count": retry_count + 1,
        }
