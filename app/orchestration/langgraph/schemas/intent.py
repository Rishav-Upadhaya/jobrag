from pydantic import BaseModel, Field

class QueryEvaluation(BaseModel):
    intent: str = Field(description="The intent of the query: valid, vague, or off_topic.")
    clarification_question: str = Field(default="", description="Question to clarify the vague intent.")
