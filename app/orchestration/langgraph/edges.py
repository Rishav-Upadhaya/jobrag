
import logging
from typing import Literal

from app.core.config import settings
from app.orchestration.langgraph.state.graph_state import GraphState

logger = logging.getLogger(__name__)


def route_intent(state: GraphState) -> Literal["reasoning_agent", "clarify", "reject"]:
    intent = state.get("intent", "off_topic")

    if intent == "valid":
        logger.info("Intent: valid → routing to reasoning_agent")
        return "reasoning_agent"
    elif intent == "vague":
        logger.info("Intent: vague → routing to clarify")
        return "clarify"
    else:  # off_topic or unknown
        logger.info("Intent: off_topic → routing to reject")
        return "reject"


def route_judge(state: GraphState) -> Literal["output", "reasoning_agent"]:
    judge_verdict = state.get("judge_verdict", "fail")
    retry_count = state.get("retry_count", 0)
    max_retries = settings.JUDGE_MAX_RETRIES

    if judge_verdict == "pass":
        logger.info("Judge verdict: PASS → outputting answer")
        return "output"

    if judge_verdict == "fail" and retry_count < max_retries:
        logger.info(f"Judge verdict: FAIL → retrying reasoning_agent (attempt {retry_count}/{max_retries})")
        return "reasoning_agent"

    logger.info(f"Judge verdict: FAIL but max retries reached → outputting answer")
    return "output"
