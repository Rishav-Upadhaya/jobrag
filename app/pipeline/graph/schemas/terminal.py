"""Terminal node schema."""

from typing import Any

from pydantic import BaseModel


class TerminalOutput(BaseModel):
    """
    Structured output from terminal nodes.

    Attributes:
        final_response: the full API response payload produced by the terminal node
    """

    final_response: dict[str, Any]
