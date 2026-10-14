# app/schemas/ticket.py

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TicketClassification(BaseModel):
    query_understood: bool

    category: str
    subcategory: str

    user_intent: str
    summary: str

    urgency: Literal["low", "medium", "high", "critical"]
    priority: Literal["low", "normal", "high", "urgent"]

    impact: str
    reason: str

    requires_human_review: bool
    action_ready: bool

    missing_information: list[str] = Field(default_factory=list)
    extracted_entities: dict[str, str] = Field(default_factory=dict)

    recommended_action: str
    next_route: str

    clarification_question: str | None = None

    verification_requirements: list[str] = Field(
        default_factory=list
    )

    resolution_condition: str

    classification_confidence: Literal[
        "low", "medium", "high"
    ]

    @model_validator(mode="after")
    def validate_action_readiness(self):
        if self.missing_information and self.action_ready:
            raise ValueError(
                "Cannot mark action_ready=True "
                "while required information is missing"
            )

        if (
            self.missing_information
            and not self.clarification_question
        ):
            raise ValueError(
                "A clarification question is required "
                "when information is missing"
            )

        return self