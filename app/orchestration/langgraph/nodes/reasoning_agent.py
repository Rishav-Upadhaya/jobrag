import logging
from typing import Any

from langchain.messages import HumanMessage
from app.infrastructure.llm.gemini_client import get_llm
from app.orchestration.langgraph.state.graph_state import GraphState
from app.orchestration.langgraph.schemas.reasoning import ReasoningOutput
from app.orchestration.langgraph.utils.messages import build_history, extract_user_query
from app.orchestration.langgraph.utils.json import parse_json_object
from app.orchestration.langgraph.prompts.reasoning_agent import REASONING_AGENT_PROMPT

logger = logging.getLogger(__name__)

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
    if not isinstance(raw, dict):
        return {}
    return {
        k: v
        for k, v in raw.items()
        if k in _VALID_FILTER_KEYS and v and str(v).strip()
    }


def reasoning_agent(state: GraphState) -> dict[str, Any]:
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

    try:
        history = (
            build_history(messages, limit=4)
            if messages
            else "No previous conversation."
        )

        formatted_prompt = REASONING_AGENT_PROMPT.format(
            query=user_query,
            history=history,
        )

        llm = get_llm("classifier", max_tokens=768)
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        structured_response = parse_json_object(content)

        top_k = structured_response.get("top_k", 5)
        top_k = max(1, min(int(top_k), 50))

        raw_filters = structured_response.get("filters", {})
        filters = _sanitize_filters(raw_filters)

        retriever_query = str(
            structured_response.get("retriever_query", user_query)
        ).strip()
        if not retriever_query:
            retriever_query = user_query

        reasoning = str(structured_response.get("reasoning", "")).strip()

        reasoning_output = ReasoningOutput(
            top_k=top_k,
            filters=filters,
            retriever_query=retriever_query,
            reasoning=reasoning,
        )

        return {
            "reasoning_output": reasoning_output,
            "retry_count": retry_count + 1,
            "resolved_query": retriever_query,
        }

    except Exception as exc:
        logger.exception("Error in reasoning_agent")
        return {
            "reasoning_output": ReasoningOutput(
                top_k=5,
                filters={},
                retriever_query=user_query,
                reasoning="Error during reasoning",
            ),
            "retry_count": retry_count + 1,
        }
