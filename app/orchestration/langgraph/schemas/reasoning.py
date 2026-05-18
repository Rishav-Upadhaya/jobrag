
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ReasoningOutput(BaseModel):

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
