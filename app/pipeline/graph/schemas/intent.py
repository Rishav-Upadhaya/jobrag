"""Intent classification schema."""

from typing import Literal

from pydantic import BaseModel


class QueryEvaluation(BaseModel):
    """
    Structured output from intent classifier node.
    
    Attributes:
        intent: Classification of the user query intent
        clarification_question: Question for user if vague query
        resolved_query: Standalone searchable query
    """

    intent: Literal["valid", "vague", "off_topic"]
    clarification_question: str = ""
    resolved_query: str = ""
    user_query: str = ""
