"""
Terminal nodes for the LangGraph pipeline.

Terminal nodes produce the final API response and do not modify graph state further.
"""

import logging
from typing import Any

from app.pipeline.graph.state.graph_state import GraphState
from app.pipeline.graph.schemas.terminal import TerminalOutput

logger = logging.getLogger(__name__)


def output_node(state: GraphState) -> TerminalOutput:
	"""
	Output terminal node.

	Builds the final API response from completed graph state.

	Args:
		state: Current graph state with all fields populated

	Returns:
		Dict with key:
		- final_response: Complete response object for API
	"""
	query = state["query"]
	answer = state.get("answer", "")
	sources = state.get("sources", [])
	judge_score = state.get("judge_score", 0.0)
	judge_verdict = state.get("judge_verdict", "fail")
	judge_reasoning = state.get("judge_reasoning", "")
	relevance_score = state.get("relevance_score", 0.0)
	quality_score = state.get("quality_score", 0.0)
	hallucination_flag = state.get("hallucination_flag", True)
	hallucination_evidence = state.get("hallucination_evidence", "none")
	intent = state.get("intent", "valid")

	final_response = {
		"query": query,
		"answer": answer,
		"sources": sources,
		"judge": {
			"score": judge_score,
			"verdict": judge_verdict,
			"reasoning": judge_reasoning,
			"relevance": relevance_score,
			"quality": quality_score,
			"hallucination": hallucination_flag,
			"hallucination_evidence": hallucination_evidence,
		},
		"intent": intent,
	}

	logger.info(
		f"Building final response: answer_length={len(answer)}, "
		f"sources_count={len(sources)}, verdict={judge_verdict}"
	)

	return {"final_response": final_response}


def clarify_node(state: GraphState) -> TerminalOutput:
	"""
	Clarify terminal node.

	Handles vague queries by requesting clarification from the user.

	Args:
		state: Current graph state with intent="vague"

	Returns:
		Dict with key:
		- final_response: Response asking for clarification
	"""
	query = state["query"]
	clarification_question = state.get("clarification_question", "")
	intent = state.get("intent", "vague")

	final_response = {
		"query": query,
		"intent": intent,
		"status": "clarification_needed",
		"clarification_question": clarification_question,
		"message": "Your query is too vague to retrieve meaningful results. "
		"Please provide more details.",
	}

	logger.info(
		f"Clarify node: requesting clarification for vague query: {query}"
	)

	return {"final_response": final_response}


def reject_node(state: GraphState) -> TerminalOutput:
	"""
	Reject terminal node.

	Handles off-topic queries by explaining they are not related to job search.

	Args:
		state: Current graph state with intent="off_topic"

	Returns:
		Dict with key:
		- final_response: Response rejecting the off-topic query
	"""
	query = state["query"]
	intent = state.get("intent", "off_topic")

	final_response = {
		"query": query,
		"intent": intent,
		"status": "off_topic",
		"message": "Your query is not related to job search, careers, or employment. "
		"Please ask a job-related question.",
	}

	logger.info(
		f"Reject node: rejecting off-topic query: {query}"
	)

	return {"final_response": final_response}
