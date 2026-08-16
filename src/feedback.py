"""User feedback logging - a thumbs up/down on each answer.

Same shape as src/observability.py's request log (one JSON line per
event, thread-safe append), kept as a separate file rather than folded
into the observability log: observability records every /api/ask call
automatically and is about system health (latency, cost, refusal rate),
while feedback is a sparse, user-initiated judgment about answer
quality - mixing the two would make "how many people actually rated
an answer" and "how many requests happened" the same denominator when
they aren't.
"""
import json
import threading
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"
_lock = threading.Lock()


def log_feedback(entry):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log():
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
