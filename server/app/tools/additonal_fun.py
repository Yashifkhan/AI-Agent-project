from app.graph.state import TicketAgentState

def extract_text(tool_result) -> str:
    """Tool ka result alag formats mein aa sakta hai, sabko text bana do."""
    if isinstance(tool_result, str):
        return tool_result
    if isinstance(tool_result, list):
        parts = []
        for item in tool_result:
            if isinstance(item, dict):
                parts.append(item.get("text") or str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(tool_result, dict):
        return tool_result.get("text") or str(tool_result)
    return str(tool_result)



def generate_answer(state: TicketAgentState):
    results = state.get("tool_results", [])
    if not results:
        return {"answer": "Koi result nahi mila."}

    if len(results) == 1:
        return {"answer": results[0]["result"]}

    # multiple sources: sabko jod do (baad me LLM se merge kara sakte ho)
    combined = "\n\n".join(
        f"[{r['source']}]\n{r['result']}" for r in results
    )
    print("combined",combined)
    return {"answer": combined}