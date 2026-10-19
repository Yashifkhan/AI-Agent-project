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
