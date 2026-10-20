from langgraph.graph import StateGraph, START, END
from app.graph.state import TicketAgentState
from app.main import classify_source
from app.router import route_query   # generate_answer jahan rakha ho wahan se import
from app.tools.additonal_fun import generate_answer

def after_classify(state: TicketAgentState):
    if state.get("action") == "route":
        return "route_query"
    return END


agent = StateGraph(TicketAgentState)

agent.add_node("classify_source", classify_source)
agent.add_node("route_query", route_query)
agent.add_node("generate_answer", generate_answer)

agent.add_edge(START, "classify_source")
agent.add_conditional_edges(
    "classify_source",
    after_classify,
    {"route_query": "route_query", END: END},
)
agent.add_edge("route_query", "generate_answer")
agent.add_edge("generate_answer", END)

app = agent.compile()

result = app.invoke({
    "user_message": "hiii my name is yashif",
    "thread_id": "test-thread-001",
    "sources": [],
    "reason": "",
    "requires_student_identification": False,
    "tool_results": [],
    "answer": "",
})

# print("result -->>>", result)
print(result.get("answer"))