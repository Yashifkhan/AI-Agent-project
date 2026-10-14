
from typing import Literal
from pydantic import BaseModel, Field


class QueryRoute(BaseModel):
    sources: list[
        Literal[
            "llm_knowledge",
            "vector_db",
            "web_search",
            "school_db"
        ]
    ] = Field(
        description="Information sources required to answer the query"
    )

    reason: str = Field(
        description="Why these sources are needed"
    )

    requires_student_identification: bool = False
