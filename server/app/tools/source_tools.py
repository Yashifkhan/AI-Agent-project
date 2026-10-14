import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)






def school_db_tool(query: str) -> str:
    return f"SCHOOL_DB function called for: {query}"


def web_search_tool(query: str) -> str:
    return f"WEB_SEARCH function called for: {query}"


def vector_db_tool(query: str) -> str:
    return f"VECTOR_DB function called for: {query}"


def llm_knowledge_tool(query: str) -> str:
    """Answer general knowledge questions using Gemini."""
    try:
        response = gemini_llm.invoke(
            [
                (
                    "system",
                    "You are a helpful AI assistant. "
                    "Answer general knowledge questions accurately "
                    "in a short, clear, useful way. "
                    "Prefer 2-4 sentences. "
                    "If a question is ambiguous, ask for clarification. "
                    "Do not invent facts."
                ),
                ("human", query),
            ]
        )

        return response.content

    except Exception as e:
        print(f"Gemini error: {e}")
        return "Sorry, I couldn't generate an answer right now."
