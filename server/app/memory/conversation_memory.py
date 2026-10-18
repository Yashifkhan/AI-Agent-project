conversation_history = {}

def get_history(thread_id: str) -> list:
    return conversation_history.get(thread_id, [])

def save_message(thread_id: str, role: str, content: str) -> None:
    conversation_history.setdefault(thread_id, []).append({
        "role": role,
        "content": content,
    })