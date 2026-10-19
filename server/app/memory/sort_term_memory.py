from threading import Lock
from typing import List, Dict

MAX_MESSAGES = 12          # sirf last 12 messages (6 turns) rakho
MAX_CHARS_PER_MESSAGE = 500  # lambe assistant answers truncate karo

_lock = Lock()
_conversation_history: Dict[str, List[dict]] = {}
_clarification_attempts: Dict[str, int] = {}


def get_history(thread_id: str) -> List[dict]:
    """Return a COPY of the last MAX_MESSAGES messages."""
    with _lock:
        return list(_conversation_history.get(thread_id, [])[-MAX_MESSAGES:])


def save_message(thread_id: str, role: str, content: str) -> None:
    with _lock:
        history = _conversation_history.setdefault(thread_id, [])
        history.append({"role": role, "content": content})
        # memory mein bhi limit rakho
        if len(history) > MAX_MESSAGES * 2:
            del history[:-MAX_MESSAGES]


def format_history(history: List[dict]) -> str:
    """Router prompt ke liye compact text."""
    if not history:
        return "(no previous messages)"
    lines = []
    for m in history:
        text = m["content"]
        if len(text) > MAX_CHARS_PER_MESSAGE:
            text = text[:MAX_CHARS_PER_MESSAGE] + "..."
        lines.append(f"{m['role']}: {text}")
    return "\n".join(lines)


def get_attempts(thread_id: str) -> int:
    with _lock:
        return _clarification_attempts.get(thread_id, 0)


def set_attempts(thread_id: str, value: int) -> None:
    with _lock:
        _clarification_attempts[thread_id] = value


def clear_thread(thread_id: str) -> None:
    with _lock:
        _conversation_history.pop(thread_id, None)
        _clarification_attempts.pop(thread_id, None)