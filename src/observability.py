"""Request-level observability logging.

Every /api/ask call gets one JSONL line: question, pattern, retrieval
distances, tokens in/out, estimated cost at Groq's list price, latency,
refused/cache status, timestamp. This is a log to run reports against on
demand (scripts/observability_report.py), not real-time alerting - no
free equivalent of PagerDuty exists at this project's scale, and
pretending otherwise would be dishonest.
"""
import json
import threading
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "observability.jsonl"
_lock = threading.Lock()

# Groq's list pricing for openai/gpt-oss-20b (the model this project
# uses as of the emergency swap in generate.py - llama-3.1-8b-instant
# was removed from Groq's platform entirely). Confirmed via web search
# against multiple independently-agreeing sources including Groq's own
# docs domain, not guessed - stale pricing here would silently corrupt
# every cost estimate this project reports. This project runs on Groq's
# free tier, so real spend is $0 - the estimate shows what a paid-tier
# deployment would cost at the same usage pattern, which is the honest
# way to report "cost" for a project with no actual billing.
PRICE_PER_M_INPUT = 0.075
PRICE_PER_M_OUTPUT = 0.30


def estimate_cost(prompt_tokens, completion_tokens):
    return round(
        prompt_tokens / 1_000_000 * PRICE_PER_M_INPUT
        + completion_tokens / 1_000_000 * PRICE_PER_M_OUTPUT,
        6,
    )


def log_request(entry):
    """Appends one JSON line. Locked for thread-safety - /api/ask handles
    concurrent requests (see server.py's backpressure semaphore), and a
    bare open().write() from two threads at once can interleave lines."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log():
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
