"""Judge node schema."""

from typing import Literal

from pydantic import BaseModel, Field


class JudgeEvaluation(BaseModel):
    """
    Structured output from judge node.
    
    Attributes:
        relevance_score: How well retrieved chunks match query (0.0-1.0)
        quality_score: Quality of generated answer (0.0-1.0)
        hallucination_flag: Whether answer contains hallucinated facts
        hallucination_evidence: Specific hallucinated claim or 'none'
        judge_score: Overall composite score (0.0-1.0)
        judge_verdict: Pass/fail based on thresholds
        judge_reasoning: Explanation of verdict
    """

    relevance_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    hallucination_flag: bool
    hallucination_evidence: str = "none"
    judge_score: float = Field(ge=0.0, le=1.0)
    judge_verdict: Literal["pass", "fail"]
    judge_reasoning: str
