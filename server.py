"""FastAPI backend for TaxCite - replaces the earlier Streamlit UI.

Endpoints:
  GET  /                 - single-page chat UI (static/index.html)
  GET  /api/patterns     - pattern registry + descriptions for the selector
  POST /api/ask          - {question, pattern, session_id} -> answer + chunks + trace
  POST /api/clear_memory - reset a session's conversation memory

Run: .venv/Scripts/python.exe -m uvicorn server:app --port 8600
"""
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cache  # noqa: E402
import feedback  # noqa: E402
import generate  # noqa: E402
import memory  # noqa: E402
import observability  # noqa: E402
import ratelimit  # noqa: E402
from guardrails import scan_chunks  # noqa: E402
from patterns import PATTERN_INFO, PATTERNS, run  # noqa: E402

# Voyage's free tier throttles to 3 requests/minute at the account level -
# that ceiling exists regardless of how many concurrent users this app
# has, so unbounded concurrent requests would just pile up into raw 429s
# from the provider instead of a clear "busy" response from this app.
# This is backpressure, not a claim about real load-tested capacity -
# actual concurrent-load testing wouldn't prove anything new here, since
# the free-tier ceiling is already the known, documented bottleneck.
_MAX_CONCURRENT_REQUESTS = 3
_request_semaphore = threading.Semaphore(_MAX_CONCURRENT_REQUESTS)
_SEMAPHORE_WAIT_SECONDS = 2

# Memory RAG's whole mechanism is answering differently for the same
# question text depending on conversation history - caching by
# (pattern, question) alone would serve a stale answer from an unrelated
# session, which is worse than no cache at all. Every other pattern's
# answer for a given question is session-independent, so caching them is
# safe (verified by construction: only memory_rag.py reads session
# history to change its retrieval query).
_UNCACHEABLE_PATTERNS = {"memory"}

app = FastAPI(title="TaxCite", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Small talk has no corpus match, so every pattern's own refusal message
# would fire on "hi" exactly as it would on a genuinely out-of-scope
# question - correct by each pattern's own citation-enforcement logic,
# but confusing UX: a greeting and a real refusal read identically to a
# first-time user. Handled here, once, before any pattern runs - applies
# uniformly regardless of which of the thirteen patterns is selected,
# and costs no retrieval or generation call.
_CHITCHAT_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|howdy|good\s?(morning|afternoon|evening)|how'?s?\s+it\s+going|"
    r"how\s+are\s+you|what'?s\s+up|sup|thanks?|thank\s+you|thx|ty|bye|goodbye|see\s+ya)[\s!.,?]*$",
    re.IGNORECASE,
)
CHITCHAT_RESPONSE = (
    "Hi! I'm a citation-grounded research assistant over the Income-tax Act, 2025 and the "
    "Income-tax Act, 1961 - every answer cites a real Act and section, so I don't have much "
    "to say outside that. Try asking something like \"What does section 139 of the 1961 Act "
    "say?\" or pick one of the example questions on the right."
)


class AskRequest(BaseModel):
    question: str
    pattern: str = "adaptive"
    session_id: str | None = None


class ClearRequest(BaseModel):
    session_id: str


class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    pattern: str
    question: str
    answer: str
    rating: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/patterns")
def patterns():
    return {"patterns": [{"name": name, **PATTERN_INFO[name]} for name in PATTERNS]}


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"error": "empty question"}
    session_id = req.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    if _CHITCHAT_PATTERN.match(question):
        return {
            "answer": CHITCHAT_RESPONSE,
            "refused": False,
            "pattern": "chitchat",
            "trace": {},
            "sources": [],
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
            "session_id": session_id,
        }

    cacheable = req.pattern not in _UNCACHEABLE_PATTERNS
    cache_embedding = None
    if cacheable:
        cached, cache_embedding = cache.get(req.pattern, question)
        if cached is not None:
            cached["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
            cached["session_id"] = session_id
            # A fresh id per response, not reused across repeat views of
            # the same cached answer - feedback on this specific render
            # shouldn't get silently merged with feedback on an earlier
            # one just because the underlying text happens to match.
            cached["message_id"] = str(uuid.uuid4())
            observability.log_request(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pattern": req.pattern,
                    "question": question,
                    "refused": cached.get("refused"),
                    "cache_hit": True,
                    "latency_seconds": cached["elapsed_seconds"],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "distances": [],
                }
            )
            return cached

    # Checked here, not earlier - chitchat and cache hits stay free for
    # everyone regardless of how much of their window they've used, since
    # this limit exists to protect the shared, expensive Groq/Voyage
    # throttle (the same resource the semaphore below protects), not to
    # punish someone for asking a lot of questions that all happen to be
    # cheap.
    if not ratelimit.check(session_id):
        return {
            "error": (
                "You've hit this session's rate limit (20 requests per 5 minutes) - "
                "this protects the shared free-tier throttle every visitor draws from. Please wait a bit."
            )
        }

    acquired = _request_semaphore.acquire(timeout=_SEMAPHORE_WAIT_SECONDS)
    if not acquired:
        return {
            "error": (
                f"System is busy (more than {_MAX_CONCURRENT_REQUESTS} requests in flight - "
                "the hosted embedding API is throttled to 3/minute on the free tier). "
                "Please retry shortly."
            )
        }
    generate.reset_usage()
    try:
        result = run(req.pattern, question, session_id=session_id)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:  # surface upstream API failures readably, don't 500
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _request_semaphore.release()

    injection_flags = scan_chunks(result.get("chunks", []))
    if injection_flags:
        # Flag, don't block - see src/guardrails.py for why.
        print(f"[guardrails] possible prompt injection in retrieved content: {injection_flags}")

    sources, seen = [], set()
    for c in result.get("chunks", []):
        m = c["metadata"]
        key = (m.get("act_id"), m.get("section"), m.get("modality", "prose"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "act": m.get("act_title"),
                "section": m.get("section") or None,
                "modality": m.get("modality", "prose"),
                "page": m.get("page"),
                "distance": round(c.get("distance", 0.0), 3),
                "via_reference": c.get("via_reference"),
            }
        )

    response = {
        "answer": result["answer"],
        "refused": result["refused"],
        "pattern": result.get("pattern", req.pattern),
        "trace": result.get("trace", {}),
        "sources": sources,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "session_id": session_id,
        "message_id": str(uuid.uuid4()),
    }

    if cacheable:
        cache.put(req.pattern, question, response, datetime.now(timezone.utc).isoformat(), embedding=cache_embedding)

    usage = generate.get_usage_totals()
    observability.log_request(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pattern": req.pattern,
            "question": question,
            "refused": result["refused"],
            "cache_hit": False,
            "latency_seconds": response["elapsed_seconds"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "estimated_cost_usd": observability.estimate_cost(usage["prompt_tokens"], usage["completion_tokens"]),
            "distances": [round(c.get("distance", 0.0), 3) for c in result.get("chunks", [])],
        }
    )

    return response


@app.post("/api/clear_memory")
def clear_memory(req: ClearRequest):
    memory.clear(req.session_id)
    return {"ok": True}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        return {"error": "rating must be 'up' or 'down'"}
    feedback.log_feedback(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": req.message_id,
            "session_id": req.session_id,
            "pattern": req.pattern,
            "question": req.question,
            "answer": req.answer,
            "rating": req.rating,
        }
    )
    return {"ok": True}
