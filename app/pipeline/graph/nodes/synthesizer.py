"""Synthesizer node.

Generates a conversational response based on retrieved context and user query,
with context from past 3 messages.

Single Responsibility: Generate answer in conversational tone.
"""

import logging

from langchain.messages import SystemMessage
from app.llm.llm_client import get_llm
from app.pipeline.graph.state.graph_state import GraphState
from app.pipeline.graph.utils.context import build_context_block_from_chunks
from app.pipeline.graph.utils.messages import build_history, extract_user_query
from app.pipeline.graph.utils.prompt_builder import (
    build_sources_from_chunks,
    fallback_answer_from_sources,
)
from app.pipeline.graph.prompts.synthesizer import SYNTHESIZER_PROMPT

logger = logging.getLogger(__name__)


def synthesizer(state: GraphState) -> dict[str, object]:
    """
    Synthesize a conversational answer using retrieved context.

    Responsibilities:
    - Extract user query and past conversation (limit 3 messages)
    - Build context from retrieved chunks
    - Generate conversational response using LLM
    - Handle edge cases (no chunks, LLM errors)

    Args:
        state: GraphState containing messages, retrieved_chunks, query info

    Returns:
        dict with keys: answer, synthesized_answer, sources
    """
    messages = state.get("messages", [])
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    # Extract user query
    if messages:
        query = extract_user_query(messages)
        # Limit history to past 3 messages for conversational context
        history = build_history(messages, limit=3)
    else:
        query = state.get("user_query") or state.get("query", "")
        history = "No previous conversation."

    logger.info(
        f"Synthesizing answer: query_len={len(query)}, chunks={len(chunks)}, "
        f"history_msgs={len(messages)}, retry={retry_count}"
    )

    if not chunks:
        logger.warning("Synthesizer: No context chunks provided")
        fallback = fallback_answer_from_sources([])
        return {
            "answer": fallback,
            "synthesized_answer": fallback,
            "sources": [],
        }

    try:
        # Build context block and extract sources
        context_text, _ = build_context_block_from_chunks(chunks)
        sources = build_sources_from_chunks(chunks)

        # Format the synthesis prompt with query, context, and conversation history
        formatted_prompt = SYNTHESIZER_PROMPT.format(
            query=query,
            context_block=context_text,
            history=history,
        )

        # Invoke LLM for answer generation
        llm = get_llm("synthesizer")
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

        logger.info("Answer synthesized successfully")

        return {
            "answer": answer,
            "synthesized_answer": answer,
            "sources": sources,
        }

    except Exception as e:
        logger.exception(f"Error in synthesizer: {e}")
        # Generate fallback answer from sources on error
        sources = build_sources_from_chunks(chunks)
        fallback = fallback_answer_from_sources(sources)
        return {
            "answer": fallback,
            "synthesized_answer": fallback,
            "sources": sources,
        }
