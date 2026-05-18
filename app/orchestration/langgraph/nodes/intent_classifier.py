import logging

from langchain.messages import HumanMessage, SystemMessage

from app.infrastructure.llm.gemini_client import get_llm
from app.orchestration.langgraph.state.graph_state import GraphState
from app.orchestration.langgraph.schemas.intent import QueryEvaluation
from app.orchestration.langgraph.utils.messages import build_history, extract_user_query
from app.orchestration.langgraph.utils.json import parse_json_object
from app.orchestration.langgraph.prompts.intent_classifier import INTENT_CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)


def _previous_user_query(messages: list) -> str:
    for message in reversed(messages[:-1]):
        if isinstance(message, HumanMessage):
            content = message.content if isinstance(message.content, str) else str(message.content)
            if content.strip():
                return content.strip()
    return ""


def intent_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])
    if messages:
        query = extract_user_query(messages)
        history = build_history(messages, limit=4)
    else:
        query = state.get("query", "")
        history = "No previous conversation."

    try:
        previous_query = _previous_user_query(messages)

        formatted_prompt = INTENT_CLASSIFIER_PROMPT.format(
            query=query,
            history=history,
        )

        llm = get_llm("classifier", max_tokens=256)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        structured_response = parse_json_object(content)

        intent = str(structured_response.get("intent", "valid")).lower()
        if intent not in ["valid", "vague", "off_topic"]:
            intent = "valid"

        clarification_question = str(structured_response.get("clarification_question", ""))

        return {
            "intent": intent,
            "clarification_question": clarification_question,
            "user_query": query,
            "retry_count": state.get("retry_count", 0),
        }

    except Exception as exc:
        logger.exception("Error in intent_classifier")
        return {
            "intent": "valid",
            "clarification_question": "",
            "user_query": query,
            "retry_count": state.get("retry_count", 0),
        }
