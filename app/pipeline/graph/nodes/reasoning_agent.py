"""Reasoning agent node.

Understands the user query with conversation context, extracts retrieval
parameters (top_k, filters), and generates a standalone retriever query.

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


def reasoning_agent(state: GraphState) -> dict[str, Any]:
    """
    Analyze user query and conversation context to extract retrieval parameters.

    Returns only the keys this node modifies.
    
    Responsibilities:
    - Extract top_k from query (default 5, range 1-50)
    - Extract filters (job_level, job_category, job_location)
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

    logger.info(f"Reasoning agent: Analyzing query: {user_query[:60]}...")

    try:
        # Build conversation history for context (limit to 4 messages)
        history = build_history(messages, limit=4) if messages else "No previous conversation."

        # Format prompt with user query and history
        formatted_prompt = REASONING_AGENT_PROMPT.format(
            query=user_query,
            history=history,
        )

        # Get reasoning from LLM
        llm = get_llm("classifier", max_tokens=512)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        # Parse structured response
        structured_response = parse_json_object(content)
        logger.debug(f"Reasoning agent response: {structured_response}")

        # Extract and validate fields
        top_k = structured_response.get("top_k", 5)
        top_k = max(1, min(int(top_k), 50))  # Clamp to 1-50

        filters = structured_response.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        # Clean up None/empty filters
        filters = {k: v for k, v in filters.items() if v}

        retriever_query = str(structured_response.get("retriever_query", user_query)).strip()
        if not retriever_query:
            retriever_query = user_query

        reasoning = str(structured_response.get("reasoning", "")).strip()

        # Build output
        reasoning_output = ReasoningOutput(
            top_k=top_k,
            filters=filters,
            retriever_query=retriever_query,
            reasoning=reasoning,
        )

        logger.info(
            f"Reasoning agent complete: top_k={top_k}, "
            f"filters={filters}, query_len={len(retriever_query)}"
        )

        return {
            "reasoning_output": reasoning_output,
            "retry_count": retry_count + 1,
            "resolved_query": retriever_query,
        }

    except Exception as exc:
        logger.exception(f"Error in reasoning_agent: {exc}")
        # Return safe defaults on error
        return {
            "reasoning_output": ReasoningOutput(
                top_k=5,
                filters={},
                retriever_query=user_query,
                reasoning="Error during reasoning",
            ),
            "retry_count": retry_count + 1,
        }
