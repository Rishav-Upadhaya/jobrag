from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    jobs_loaded: int
    chunks_created: int
    dropped_rows: int
    time_seconds: float


class SourceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    job_title: str | None = None
    company_name: str | None = None
    job_level: str | None = None
    job_location: str | None = None
    relevance_score: float | None = None
    matched_chunk: str | None = None


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    verdict: str
    reasoning: str
    relevance: float
    quality: float
    hallucination: bool


class JudgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    hallucination_flag: bool
    hallucination_evidence: str = "none"
    judge_score: float = Field(ge=0.0, le=1.0)
    judge_verdict: Literal["pass", "fail"]
    judge_reasoning: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    judge: JudgeResult
    intent: str
    latency_ms: float
    clarification_question: str | None = None
    message: str | None = None
    status: str | None = None
