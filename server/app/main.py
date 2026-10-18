import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings,)
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.schemas.ticket import TicketClassification
from app.workflows.router import route_ticket
from app.schemas.query_route import QueryRoute
from app.router import route_query
load_dotenv()
from app.memory.conversation_memory import (
    get_history,
    save_message,
)


# NVIDIA
nvidia_llm = ChatNVIDIA(
    model="google/gemma-4-31b-it",
)

# Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

classifier = groq_llm.with_structured_output(
    TicketClassification
)

def classify_ticket(user_message: str):
    prompt = f"""
    You are an AI ticket classification system.

    Analyze the customer's message.

    Identify:
    - Category and subcategory
    - User intent and summary
    - Urgency, priority, and impact
    - Missing information
    - Recommended action and next route
    - Whether human review is required

    Rules:
    - Never claim that a payment has been verified.
    - If a student identifier is required but missing,
      include it in missing_information.
    - If required information is missing,
      set action_ready to false.
    - Ask a clarification question when information is missing.
    - Use only the information provided by the user.
    - Return only the fields defined by the schema.

    Customer message:
    {user_message}
    """

    return classifier.invoke(prompt)


# result = classify_ticket(
#     "I paid all fee but my panel shows pending fee"
# )


def main():
    user_message = "I paid all fee but my panel shows pending fee"
    ticket = classify_ticket(user_message)
    # Step 2: Route the validated classification
    route_result = route_ticket(ticket)

    print("\n--- Routing Result ---")
    print(route_result)


# if __name__ == "__main__":
#     main()

query_router_llm = groq_llm.with_structured_output(
    QueryRoute,
    # method="json_schema",
    # strict=True,
    )


def classify_source(data: dict):
    user_message = data["user"]
    # thread_id = data["thread_id"]

    print("User query analysis...")

    # history = get_history(thread_id)

    # history_text = "\n".join(
    #     f"{message['role']}: {message['content']}"
    #     for message in history
    # )

    # prompt = f"""
    # You are a query source classifier for school management software.

    # Available sources:
    # - llm_knowledge: General knowledge.
    # - vector_db: Internal school manuals and instructions.
    # - web_search: Current public information.
    # - school_db: Actual school records.

    # Conversation history:
    # {history_text if history_text else "No previous conversation."}

    # Classify the current question using the conversation history
    # when needed to understand follow-up questions.

    # Current user question:
    # {user_message}
    # """

    prompt = f"""
    You are a query source classifier for school management software.

    Available sources:

    llm_knowledge:
    General concepts that do not require private or product-specific data.

    vector_db:
    Internal school software manuals, feature instructions,
    FAQs, and approved product documentation.

    web_search:
    Current public information that needs external verification.

    school_db:
    Actual school records such as exam history, fee status,
    attendance, and student details.

  Conversation memory rules:
- If the user shares personal information or a fact intended
  to be remembered during this conversation, do not treat it
  as an information-seeking query.
- Set sources=[] when no external source or tool is needed.
- The application must save the user's message to conversation
  memory even when sources=[].
- If the user asks a general-knowledge question, select llm_knowledge.
- If the user asks a follow-up question that can be answered from
  conversation history, use that history to answer it.
- Do not route personal statements to school_db unless the user
  explicitly requests an actual school-record lookup.

    User question:
    {user_message}
    """
    
    result = query_router_llm.invoke(prompt)

    print("Classification:", result)

    response = route_query(user_message, result)

    return response




# thread_id = "conversation-101"
# query = "My test student name is Rahul Sharma."
# query = "What is my test student's name?"
# query="what is the full form **HBSE** board"
# query="what is the process of remove the student from routes in transport , and tell me latest news of hbsc board"
# query="kya kal baris ho skti hai ,m kal chutty marne ki so/ch rha hu"
# query="get the top 5 student, which have panding more fee to another student"
# query="diwali fastival holiday date and day in 2026"
# query="what si the gold price today ,22 karet gold and 24 karet gold"
# query="what is the process of new student addmission "
# query="who to check the student complete info,step by step tell me"
# result=classify_source({"user":query,"thread_id":thread_id})
# print("\n--- Query ---")
# print(result["query"])

# print("\n--- Reason ---")
# print(result["reason"])

# for item in result["results"]:
#     print(f"\n--- Answer from {item['source']} ---")
#     print(item["result"])
    
# queries = [
#     "For this conversation, remember that my favourite colour is blue.",
#     "What is my favourite colour?"
# ]


# for query in queries:
#     result = classify_source({
#         "user": query,
#         "thread_id": thread_id
#     })

#     for item in result["results"]:
#         print(f"\n--- Answer from {item['source']} ---")
#         print(item["result"])

query = "My test student name is Rahul Sharma."
# query = "what is ai/ml"
# thread_id = "memory-test-101"

result = classify_source({
    "user": query,
    # "thread_id": thread_id
})

print(result)