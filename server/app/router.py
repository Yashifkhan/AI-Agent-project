from app.tools.source_tools import (
    school_db_tool,
    web_search_tool,
    vector_db_tool,
    llm_knowledge_tool,
)

# Allow only known, registered sources.
SOURCE_TO_TOOL = {
    "school_db": school_db_tool,
    "web_search": web_search_tool,
    "vector_db": vector_db_tool,
    "llm_knowledge": llm_knowledge_tool,
}

def route_query(query: str, model_result) -> dict:
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

        result = tool(query)
        # print("tool result after query check  -->>",result[0]["text"])

        results.append({
            "source": source,
            "result": result,
        })

    return {
        "query": query,
        "reason": model_result.get("reason", ""),
        "results": results[0]["result"][0]["text"],
    }