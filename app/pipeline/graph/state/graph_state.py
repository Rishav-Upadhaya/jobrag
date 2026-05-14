"""LangGraph GraphState definition.

This is the single source of truth for state shape across all graph nodes.
Uses proper TypedDict with total=False for optional fields.
"""

from typing import Literal, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.graph import MessagesState
from app.pipeline.graph.schemas.intent import QueryEvaluation
from app.pipeline.graph.schemas.reasoning import ReasoningOutput


class GraphState(MessagesState):
    """
    Shared LangGraph state for the job search RAG pipeline.

    All fields are optional (total=False) to allow nodes to return only
    the keys they modify, per LangGraph best practices.
    """

    user_query: str
    query: str
    filters: dict
    top_k: int
    query_evaluation: QueryEvaluation
    intent_evaluation: str
    intent: str
    clarification_question: str
    resolved_query: str
    reasoning_output: ReasoningOutput
    retrieved_chunks: list[dict]
    synthesized_answer: str
    answer: str
    sources: list[dict]
    judge_score: float
    judge_verdict: Literal["pass", "fail"]
    judge_reasoning: str
    relevance_score: float
    quality_score: float
    hallucination_flag: bool
    hallucination_evidence: str
    retry_count: int
    final_response: dict
    history: list[dict]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_evaluation: QueryEvaluation
    retry_count: int
    top_k: int
    