import logging

from langchain.messages import HumanMessage
from app.infrastructure.llm.gemini_client import get_llm
from app.orchestration.langgraph.state.graph_state import GraphState
from app.orchestration.langgraph.utils.context import build_context_block_from_chunks
from app.orchestration.langgraph.utils.messages import build_history, extract_user_query
from app.orchestration.langgraph.utils.prompt_builder import (
    build_sources_from_chunks,
    fallback_answer_from_sources,
)
from app.orchestration.langgraph.prompts.synthesizer import SYNTHESIZER_PROMPT

logger = logging.getLogger(__name__)


def synthesizer(state: GraphState) -> dict[str, object]:
    messages = state.get("messages", [])
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    if messages:
        query = extract_user_query(messages)
        history = build_history(messages, limit=3)
    else:
        query = state.get("user_query") or state.get("query", "")
        history = "No previous conversation."

    if not chunks:
        logger.warning("Synthesizer: No context chunks provided")
        fallback = fallback_answer_from_sources([])
        return {
            "answer": fallback,
            "synthesized_answer": fallback,
            "sources": [],
        }

    try:
        context_text, _ = build_context_block_from_chunks(chunks)
        sources = build_sources_from_chunks(chunks)

        formatted_prompt = SYNTHESIZER_PROMPT.format(
            query=query,
            context_block=context_text,
            history=history,
        )

        llm = get_llm("synthesizer")
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

        logger.info("Answer synthesized successfully")

        return {
            "answer": answer,
            "synthesized_answer": answer,
            "sources": sources,
        }

    except Exception as e:
        logger.exception("Error in synthesizer")
        sources = build_sources_from_chunks(chunks)
        fallback = fallback_answer_from_sources(sources)
        return {
            "answer": fallback,
            "synthesized_answer": fallback,
            "sources": sources,
        }
