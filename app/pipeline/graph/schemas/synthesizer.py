"""Synthesizer node schema."""

from typing import Any, List

from pydantic import BaseModel


class SynthesizerOutput(BaseModel):
    """
    Structured output from synthesizer node.
    
    Attributes:
        answer: LLM-generated answer to user query
        synthesized_answer: canonical copy of the answer (for backwards compat)
        sources: list of source metadata dicts attached to the answer
    """

    answer: str
    synthesized_answer: str
    sources: List[dict[str, Any]] = []
