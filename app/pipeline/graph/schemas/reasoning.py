"""Reasoning agent output schema.

Structured output from reasoning_agent node containing extracted parameters
and the standalone retriever query.

Extended to support 7 filter fields (up from 3):
  - job_level, job_category, job_location  (original)
  - company_name, job_title               (new — exact/LIKE match on SQL)
  - date_order                             (new — "desc" → ORDER BY pub_date DESC)
  - date_after                             (new — ISO date string → WHERE pub_date > ?)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ReasoningOutput(BaseModel):
    """Output from reasoning agent node.

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
        description=(
            "Structured filters extracted from the query. Valid keys: "
            "job_level, job_category, job_location, company_name, job_title, "
            "date_order ('asc'|'desc'), date_after (ISO date string 'YYYY-MM-DD')."
        ),
    )
    retriever_query: str = Field(
        ...,
        description="Standalone, self-contained query for retrieval",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of what user is searching for",
    )
