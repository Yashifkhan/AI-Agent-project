from typing import Literal, List
from pydantic import BaseModel, Field, model_validator

class QueryRoute(BaseModel):
    action: Literal[
        "route",
        "ask_clarification",
        "rephrase_and_ask",
        "cannot_answer",
    ] = Field(
        description=(
            "route: query understood and sources selected. "
            "ask_clarification: query unclear, ask user to repeat. "
            "rephrase_and_ask: still unclear, ask in simple words. "
            "cannot_answer: final fallback after retries."
        )
    )

    sources: List[
        Literal["llm_knowledge", "vector_db", "web_search", "school_db"]
    ] = Field(
        default_factory=list,
        description="Sources needed. Must be non-empty when action is 'route'; empty otherwise.",
    )

    # requires_student_identification: bool = Field(
    #     default=False,
    #     description="True only if the query needs a specific student identified.",
    # )

    reason: str = Field(
        description="One short sentence specific to THIS message. Do not copy example text."
    )

    message_to_user: str = Field(
        default="",
        description="Text shown to the user for non-route actions; empty string for route.",
    )

    @model_validator(mode="after")
    def check_consistency(self):
        if self.action == "route" and not self.sources:
            # Safety net: conversational/unknown queries fall back to llm_knowledge
            self.sources = ["llm_knowledge"]
        if self.action != "route":
            self.sources = []
        return self

