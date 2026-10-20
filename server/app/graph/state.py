
from typing import Any, TypedDict,List


class TicketAgentState(TypedDict, total=False):
    user_message: str
    thread_id: str
    history: List[dict]
    model_result: Any
    action: str
    message_to_user: str
    sources: List[str]
    reason: str
    requires_student_identification: bool
    tool_results: List[dict]
    answer: str

# class TicketAgentState(TypedDict):
#     # Input
#     user_message: str
#     thread_id: str

#     # Classification
#     sources: list[str]
#     reason: str
#     requires_student_identification: bool

#     # Tool execution
#     tool_results: list[dict[str, Any]]

#     # Final response
#     answer: str
