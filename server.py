"""FastAPI backend for TaxCite - replaces the earlier Streamlit UI.

Endpoints:
  GET  /                 - single-page chat UI (static/index.html)
  GET  /api/patterns     - pattern registry + descriptions for the selector
  POST /api/ask          - {question, pattern, session_id} -> answer + chunks + trace
  POST /api/clear_memory - reset a session's conversation memory

Run: .venv/Scripts/python.exe -m uvicorn server:app --port 8600
"""
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import memory  # noqa: E402
from patterns import PATTERN_INFO, PATTERNS, run  # noqa: E402

app = FastAPI(title="TaxCite", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    question: str
    pattern: str = "adaptive"
    session_id: str | None = None


class ClearRequest(BaseModel):
    session_id: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/patterns")
def patterns():
    return {"patterns": [{"name": name, "description": PATTERN_INFO[name]} for name in PATTERNS]}


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"error": "empty question"}
    session_id = req.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        result = run(req.pattern, question, session_id=session_id)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:  # surface upstream API failures readably, don't 500
        return {"error": f"{type(e).__name__}: {e}"}

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

    return {
        "answer": result["answer"],
        "refused": result["refused"],
        "pattern": result.get("pattern", req.pattern),
        "trace": result.get("trace", {}),
        "sources": sources,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "session_id": session_id,
    }


@app.post("/api/clear_memory")
def clear_memory(req: ClearRequest):
    memory.clear(req.session_id)
    return {"ok": True}
