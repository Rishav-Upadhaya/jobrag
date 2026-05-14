"""Reasoning agent output schema.

Structured output from reasoning_agent node containing extracted parameters
and the standalone retriever query.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ReasoningOutput(BaseModel):
    """
    Output from reasoning agent node.
    
    This node understands the user query with conversation context,
    extracts retrieval parameters, and generates a standalone query.
    """

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of jobs to retrieve (1-50)",
    )
    filters: dict = Field(
        default_factory=dict,
        description="Structured filters: {job_level?, job_category?, job_location?}",
    )
    retriever_query: str = Field(
        ...,
        description="Standalone, self-contained query for retrieval",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of what user is searching for",
    )
