from app.tools.source_tools import (
    school_db_tool,
    web_search_tool,
    vector_db_tool,
    llm_knowledge_tool,
)
from app.memory.sort_term_memory import format_history


# Allow only known, registered sources.
SOURCE_TO_TOOL = {
    "school_db": school_db_tool,
    "web_search": web_search_tool,
    "vector_db": vector_db_tool,
    "llm_knowledge": llm_knowledge_tool,
}

def route_query(query: str, model_result, history: list | None = None):
    sources = model_result.sources
    reason = model_result.reason

    invalid_sources = [s for s in sources if s not in SOURCE_TO_TOOL]
    if invalid_sources:
        raise ValueError(f"Unknown sources: {invalid_sources}")

    results = []

    for source in sources:
        print(f"Routing to: {source}")
        tool = SOURCE_TO_TOOL[source]

        # Sirf llm_knowledge ko conversation context do
        if source == "llm_knowledge" and history:
            tool_query = (
                "Previous conversation:\n"
                f"{format_history(history)}\n\n"
                f"Current user message: {query}\n\n"
                "Answer the current message using the previous conversation "
                "when it is relevant."
            )
        else:
            tool_query = query

        try:
            tool_result = tool(tool_query)
        except Exception as e:
            print(f"Tool error ({source}): {e}")
            tool_result = "This source could not provide a result."

        results.append({"source": source, "result": tool_result})

    return {"query": query, "reason": reason, "results": results}

# def route_query(query: str, model_result):
    sources = model_result.sources
    reason = model_result.reason

    invalid_sources = [
        source for source in sources
        if source not in SOURCE_TO_TOOL
    ]

    if invalid_sources:
        raise ValueError(f"Unknown sources: {invalid_sources}")

    results = []

    for source in sources:
        print("llm select the source -->>",source)
        tool = SOURCE_TO_TOOL[source]
        print(f"Routing to: {source}")

        try:
            tool_result = tool(query)
            print("tool_result")
        except Exception as e:
            print(f"Tool error ({source}): {e}")
            tool_result = "This source could not provide a result."

        results.append({
            "source": source,
            "result": tool_result,
        })

    return {
        "query": query,
        "reason": reason,
        "results": results,
    }


# def route_query(query: str, model_result) -> dict:
    print("pass query to correct route ...")

    # Convert Pydantic model to a Python dictionary.
    if hasattr(model_result, "model_dump"):
        model_result = model_result.model_dump()

    sources = model_result.get("sources", [])

    # print("Selected sources:", sources)

    if not isinstance(sources, list) or not sources:
        raise ValueError("No valid sources selected")

    # Validate all sources before executing any function.
    invalid_sources = [
        source for source in sources
        if source not in SOURCE_TO_TOOL
    ]

    if invalid_sources:
        raise ValueError(f"Unknown sources: {invalid_sources}")

    results = []

    for source in sources:        
        tool = SOURCE_TO_TOOL[source]
        print(f"Routing to: {source}")
        
        # tool calling 
        result = tool(query)
        # print("tool result after query check  -->>",result[0]["text"])

        results.append({
            "source": source,
            "result": result,
        })

    return {
        "query": query,
        "reason": model_result.get("reason", ""),
        "results": results
    }