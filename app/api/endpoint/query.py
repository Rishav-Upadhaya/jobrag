from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain.messages import HumanMessage, AIMessage

from app.models.request import QueryRequest
from app.models.response import JudgeResult, QueryResponse, SourceItem
from app.pipeline.graph.graph import build_graph

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_judge_result(raw: dict[str, Any] | None) -> JudgeResult:
	raw = raw or {}
	return JudgeResult(
		score=float(raw.get("score", 0.0)),
		verdict=str(raw.get("verdict", "fail")),
		reasoning=str(raw.get("reasoning", "")),
		relevance=float(raw.get("relevance", 0.0)),
		quality=float(raw.get("quality", 0.0)),
		hallucination=bool(raw.get("hallucination", False)),
	)


def _build_sources(raw_sources: list[dict[str, Any]], top_k: int) -> list[SourceItem]:
	sources = [
		SourceItem(
			job_id=str(source.get("job_id", "")),
			job_title=source.get("job_title"),
			company_name=source.get("company_name"),
			job_level=source.get("job_level"),
			job_location=source.get("job_location"),
			relevance_score=source.get("relevance_score"),
			matched_chunk=source.get("matched_chunk"),
		)
		for source in raw_sources
	]
	return sources[:top_k]


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
	json_string = json.dumps(payload, ensure_ascii=True)
	return f"event: {event_name}\ndata: {json_string}\n\n"


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
	"""
	Main query endpoint.
	
	The backend now handles parameter extraction:
	- reasoning_agent extracts top_k and filters from user query + conversation
	- Frontend only sends query and conversation_history
	"""
	start = time.monotonic()

	user_query = req.query.strip()
	if not user_query:
		raise HTTPException(status_code=400, detail="Query cannot be empty")

	# Build conversation history from prior messages
	conversation_history = []
	for msg in req.conversation_history:
		role = msg.get("role", "user").lower()
		content = str(msg.get("content", ""))
		if role == "user":
			conversation_history.append(HumanMessage(content=content))
		elif role == "assistant":
			conversation_history.append(AIMessage(content=content))
	
	# Append current user query
	conversation_history.append(HumanMessage(content=user_query))

	try:
		# Build graph and invoke with only messages and query
		# top_k and filters are extracted by reasoning_agent
		graph = build_graph()
		initial_state: dict[str, Any] = {
			"messages": conversation_history,
			"query": user_query,
		}

		result = await graph.ainvoke(initial_state)
	except Exception as exc:
		logger.exception("Error invoking graph")
		raise HTTPException(status_code=500, detail="Internal error during query processing") from exc

	if hasattr(result, "model_dump") and callable(getattr(result, "model_dump")):
		try:
			completed_state = result.model_dump()
		except TypeError:
			completed_state = result.model_dump(exclude_none=True)
	elif isinstance(result, dict):
		completed_state = result
	else:
		logger.warning("Graph returned non-mapping result: %s", type(result))
		completed_state = {}

	final_response = completed_state.get("final_response") or completed_state
	latency_ms = (time.monotonic() - start) * 1000.0
	judge = _build_judge_result(final_response.get("judge"))
	final_sources = final_response.get("sources", [])
	sources = _build_sources(final_sources, len(final_sources) or 5)
	answer = final_response.get("answer")
	if answer == "":
		answer = None

	return QueryResponse(
		query=str(final_response.get("query", req.query)),
		answer=answer,
		sources=sources,
		judge=judge,
		intent=str(final_response.get("intent", "valid")),
		latency_ms=latency_ms,
		clarification_question=final_response.get("clarification_question"),
		message=final_response.get("message"),
		status=final_response.get("status"),
	)


@router.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest) -> StreamingResponse:
	"""
	Streaming endpoint that sends SSE events during query processing.
	
	Backend handles parameter extraction via reasoning_agent.
	"""

	async def generate_stream():
		start = time.monotonic()

		user_query = req.query.strip()
		if not user_query:
			yield _format_sse("error", {"message": "Query cannot be empty"})
			return

		# Build conversation history from prior messages
		conversation_history = []
		for msg in req.conversation_history:
			role = msg.get("role", "user").lower()
			content = str(msg.get("content", ""))
			if role == "user":
				conversation_history.append(HumanMessage(content=content))
			elif role == "assistant":
				conversation_history.append(AIMessage(content=content))
		
		# Append current user query
		conversation_history.append(HumanMessage(content=user_query))

		try:
			yield _format_sse("status", {"stage": "classifying", "message": "Analyzing your query..."})

			# Build graph and invoke with only messages and query
			graph = build_graph()
			initial_state: dict[str, Any] = {
				"messages": conversation_history,
				"query": user_query,
			}

			result = await graph.ainvoke(initial_state)

			if hasattr(result, "model_dump") and callable(getattr(result, "model_dump")):
				try:
					completed_state = result.model_dump()
				except TypeError:
					completed_state = result.model_dump(exclude_none=True)
			elif isinstance(result, dict):
				completed_state = result
			else:
				logger.warning("Graph returned non-mapping result: %s", type(result))
				completed_state = {}

			final_response = completed_state.get("final_response") or completed_state
			latency_ms = (time.monotonic() - start) * 1000.0

			# Send answer
			answer_text = final_response.get("answer")
			intent = str(final_response.get("intent", "valid"))
			clarification_question = final_response.get("clarification_question")
			if intent == "vague" and clarification_question:
				answer_text = clarification_question
			elif not answer_text:
				answer_text = final_response.get("message") or final_response.get("clarification_question") or ""

			yield _format_sse("answer", {"text": str(answer_text)})

			# Send sources and judge results
			final_sources = final_response.get("sources", [])
			sources = _build_sources(final_sources, len(final_sources) or 5)
			judge = _build_judge_result(final_response.get("judge"))

			yield _format_sse(
				"sources",
				{
					"sources": [source.model_dump() for source in sources],
					"judge": judge.model_dump(),
					"intent": intent,
					"clarification_question": clarification_question,
					"latency_ms": latency_ms,
				},
			)

			yield _format_sse("done", {})

		except Exception as exc:
			logger.exception("Error in query stream")
			yield _format_sse("error", {"message": f"Error processing query: {str(exc)}"})


	return StreamingResponse(generate_stream(), media_type="text/event-stream")
