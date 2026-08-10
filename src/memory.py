"""In-process conversation memory (RAG-with-memory pattern).

Deliberately simple: a per-session ring buffer of the last N turns, held
in process memory. No persistence across server restarts - for a
portfolio demo the interesting part is how memory changes *retrieval*
(follow-up questions like "and under the old Act?" are unanswerable
without it), not durable storage. Swapping in Redis/SQLite later wouldn't
change the pattern's shape.
"""
from collections import defaultdict, deque

MAX_TURNS = 6  # question+answer pairs kept per session

_sessions = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


def add_turn(session_id, question, answer):
    _sessions[session_id].append(("user", question))
    _sessions[session_id].append(("assistant", answer))


def get_history(session_id):
    return list(_sessions.get(session_id, []))


def format_history(session_id, max_chars=2000):
    """History as plain text for prompt inclusion, newest-truncated."""
    lines = [f"{role}: {text}" for role, text in get_history(session_id)]
    joined = "\n".join(lines)
    return joined[-max_chars:] if len(joined) > max_chars else joined


def clear(session_id):
    _sessions.pop(session_id, None)
