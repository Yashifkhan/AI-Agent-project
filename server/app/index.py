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
agent.add_conditional_edges( "classify_source",  after_classify,
                            {"route_query": "route_query", END: END},
)
agent.add_edge("route_query", "generate_answer")
agent.add_edge("generate_answer", END)

app = agent.compile()

result = app.invoke({
    "user_message": "my",
    "thread_id": "test-thread-001",
    "sources": [],
    "reason": "",
    "requires_student_identification": False,
    "tool_results": [],
    "answer": "",
})



import uuid

app = agent.compile()

def chat_loop():
    thread_id = str(uuid.uuid4())
    print("Chat started. Type 'stop' or 'exit' to end.\n")

    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue

        if user_text.lower() in ("stop", "exit"):
            print("AI: Goodbye!")
            break

        try:
            result = app.invoke({
                "user_message": user_text,
                "thread_id": thread_id,
                "sources": [],
                "reason": "",
                "requires_student_identification": False,
                "tool_results": [],
                "answer": "",
            })

            print(f"AI: {result.get('answer', 'No answer returned.')}\n")

        except Exception as e:
            print(f"AI: Something went wrong: {e}\n")

if __name__ == "__main__":
    chat_loop()

# print("result -->>>", result)
# print(result.get("answer"))