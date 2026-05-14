from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class QueryRequest(BaseModel):
    """
    Request model for /query endpoint.
    
    Note: top_k and filters are NO LONGER accepted.
    The backend reasoning_agent extracts these from the user query instead.
    """
    session_id: str | None = None
    query: str = Field(..., description="User search query")
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Previous messages: [{role: user/assistant, content: str}, ...]"
    )
    model_config = ConfigDict(extra="forbid")
