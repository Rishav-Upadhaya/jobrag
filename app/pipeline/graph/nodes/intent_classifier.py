"""Intent classifier node.

Classifies the user query intent: valid, vague, or off_topic.

Single Responsibility: Intent classification only.
Does NOT extract filters, top_k, or generate retriever query.
That is handled by the reasoning_agent node.
"""

import logging

from langchain.messages import HumanMessage, SystemMessage

from app.llm.llm_client import get_llm
from app.pipeline.graph.state.graph_state import GraphState
from app.pipeline.graph.schemas.intent import QueryEvaluation
from app.pipeline.graph.utils.messages import build_history, extract_user_query
from app.pipeline.graph.utils.json import parse_json_object
from app.pipeline.graph.prompts.intent_classifier import INTENT_CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)


def _previous_user_query(messages: list) -> str:
    """Extract the previous (not current) user query from messages."""
    for message in reversed(messages[:-1]):
        if isinstance(message, HumanMessage):
            content = message.content if isinstance(message.content, str) else str(message.content)
            if content.strip():
                return content.strip()
    return ""


def intent_classifier(state: GraphState) -> dict:
    """
    Classify the user query intent: valid, vague, or off_topic.

    Single responsibility: Determine if query is valid/vague/off_topic.
    Parameter extraction (top_k, filters, retriever_query) is delegated
    to the reasoning_agent node.

    Returns only the keys this node modifies.
    """

    logger.info("Intent classifier: Starting classification")

    messages = state.get("messages", [])
    if messages:
        query = extract_user_query(messages)
        history = build_history(messages, limit=4)
    else:
        query = state.get("query", "")
        history = "No previous conversation."

    logger.info(f"Classifying intent for query: {query[:50]}...")

    try:
        previous_query = _previous_user_query(messages)

        # Format prompt with query and history
        formatted_prompt = INTENT_CLASSIFIER_PROMPT.format(
            query=query,
            history=history,
        )

        # Get classification from LLM
        llm = get_llm("classifier", max_tokens=256)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        structured_response = parse_json_object(content)

        logger.debug(f"Intent classification response: {structured_response}")

        # Extract intent and clarification question
        intent = str(structured_response.get("intent", "valid")).lower()
        if intent not in ["valid", "vague", "off_topic"]:
            intent = "valid"

        clarification_question = str(structured_response.get("clarification_question", ""))

        logger.info(f"Intent classified as: {intent}")

        return {
            "intent": intent,
            "clarification_question": clarification_question,
            "user_query": query,
            "retry_count": state.get("retry_count", 0),
        }

    except Exception as exc:
        logger.exception(f"Error in intent_classifier: {exc}")
        # Default to valid and let reasoning_agent handle it
        return {
            "intent": "valid",
            "clarification_question": "",
            "user_query": query,
            "retry_count": state.get("retry_count", 0),
        }
