from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class QueryRequest(BaseModel):
    session_id: str | None = None
    query: str = Field(..., description="User search query")
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Previous messages: [{role: user/assistant, content: str}, ...]"
    )
    model_config = ConfigDict(extra="forbid")
