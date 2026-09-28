import logging
import time

from langchain.messages import HumanMessage
from app.core.config import settings
from app.orchestration.langgraph.state.graph_state import GraphState
from app.orchestration.langgraph.utils.context import build_context_block_from_chunks
from app.orchestration.langgraph.utils.json import clamp_score, parse_json_object
from app.orchestration.langgraph.prompts.judge import JUDGE_PROMPT
from app.infrastructure.llm.gemini_client import get_llm

logger = logging.getLogger(__name__)

def _is_no_match_answer(answer: str) -> bool:
    return "no matching jobs found" in answer.lower()


def judge(state: GraphState) -> dict[str, object]:
    query = state.get("resolved_query") or state.get("user_query") or state.get("query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    answer = state.get("synthesized_answer") or state.get("answer", "")
    retry_count = state.get("retry_count", 0)

    if not answer:
        logger.warning("No answer to judge")
        return {
            "judge_score": 0.0,
            "judge_verdict": "fail",
            "judge_reasoning": "No answer was generated.",
            "relevance_score": 0.0,
            "quality_score": 0.0,
            "hallucination_flag": True,
            "hallucination_evidence": "none",
        }

    if not retrieved_chunks:
        no_match = _is_no_match_answer(answer)
        verdict = "pass" if no_match else "fail"
        return {
            "judge_score": 1.0 if no_match else 0.0,
            "judge_verdict": verdict,
            "judge_reasoning": "No retrieved jobs; no-match response is appropriate." if no_match else "Answer was generated without retrieved jobs.",
            "relevance_score": 1.0 if no_match else 0.0,
            "quality_score": 1.0 if no_match else 0.0,
            "hallucination_flag": not no_match,
            "hallucination_evidence": "none" if no_match else "answer without retrieved context",
        }

    try:
        context_block, chunks_included = build_context_block_from_chunks(
            retrieved_chunks,
            max_tokens=4000,
        )

        formatted_prompt = JUDGE_PROMPT.format(
            query=query,
            context_block=context_block,
            answer=answer,
            threshold=settings.JUDGE_PASS_THRESHOLD,
        )

        llm = get_llm("judge", max_tokens=512)
        max_retries = 3
        backoff = 0.5
        last_exc: Exception | None = None
        content = None
        for attempt in range(1, max_retries + 1):
            try:
                response = llm.invoke([HumanMessage(content=formatted_prompt)])
                content = response.content if hasattr(response, "content") else str(response)
                break
            except Exception as e:
                logger.warning("LLM error on attempt %s/%s", attempt, max_retries)
                last_exc = e

            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

        if not content:
            raise last_exc or ValueError("No response from LLM after retries")

        structured_response = parse_json_object(content)
        if not structured_response:
            logger.warning("Judge LLM returned non-JSON or empty response")
            return {
                "relevance_score": 0.7,
                "quality_score": 0.7,
                "hallucination_flag": False,
                "hallucination_evidence": "none",
                "judge_score": 0.7,
                "judge_verdict": "pass",
                "judge_reasoning": "Judge unavailable; allowing grounded retrieval result without retry.",
            }

        relevance_score = clamp_score(structured_response.get("relevance_score", 0.0))
        quality_score = clamp_score(structured_response.get("quality_score", 0.0))
        hallucination_flag = bool(structured_response.get("hallucination_flag", False))
        hallucination_evidence = str(structured_response.get("hallucination_evidence", "none"))
        judge_reasoning = str(structured_response.get("judge_reasoning", ""))
        judge_score = (relevance_score + quality_score) / 2.0

        threshold = settings.JUDGE_PASS_THRESHOLD
        judge_verdict = "pass" if judge_score >= threshold and not hallucination_flag else "fail"

        logger.info("Judge verdict=%s score=%.2f threshold=%.2f", judge_verdict, judge_score, threshold)

        return {
            "relevance_score": relevance_score,
            "quality_score": quality_score,
            "hallucination_flag": hallucination_flag,
            "hallucination_evidence": hallucination_evidence,
            "judge_score": judge_score,
            "judge_verdict": judge_verdict,
            "judge_reasoning": judge_reasoning,
        }

    except Exception as exc:
        logger.exception("Error in judge")
        return {
            "relevance_score": 0.7,
            "quality_score": 0.7,
            "hallucination_flag": False,
            "hallucination_evidence": "none",
            "judge_score": 0.7,
            "judge_verdict": "pass",
            "judge_reasoning": "Judge unavailable; allowing grounded retrieval result without retry.",
        }
