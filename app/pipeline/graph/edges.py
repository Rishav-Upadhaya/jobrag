"""
Conditional edge routing functions for the LangGraph pipeline.

These functions determine which node to execute next based on the current state.
Single Responsibility: Each function routes based on one evaluation criterion.
"""

import logging
from typing import Literal

from app.config import settings
from app.pipeline.graph.state.graph_state import GraphState

logger = logging.getLogger(__name__)


def route_intent(state: GraphState) -> Literal["reasoning_agent", "clarify", "reject"]:
    """
    Route based on intent classification result.

    Single Responsibility: Route based on intent (valid/vague/off_topic).

    Args:
        state: Current graph state with intent field

    Returns:
        - "reasoning_agent" if intent is "valid" (proceed to parameter extraction)
        - "clarify" if intent is "vague" (ask user for clarification)
        - "reject" if intent is "off_topic" (reject the query)
    """
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
    """
    Route based on judge evaluation result.

    Single Responsibility: Determine if answer passes validation or should be retried.

    Strategy:
    - If judge verdict is "pass": output answer
    - If judge verdict is "fail" and retry_count < max_retries:
      Retry the reasoning_agent to re-understand the query and extract better parameters
    - Otherwise: output answer regardless

    Args:
        state: Current graph state with judge_verdict and retry_count

    Returns:
        - "output" if judge passed or max retries reached
        - "reasoning_agent" if judge failed and retries remaining
    """
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
