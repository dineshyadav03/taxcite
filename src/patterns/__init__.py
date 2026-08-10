"""Registry of the ten RAG patterns.

Every pattern is a callable answer(question, session_id=None) returning:
    {
      "answer":  str,
      "chunks":  [retrieved chunk dicts],
      "refused": bool,
      "trace":   {pattern-specific record of what actually happened},
      "pattern": str,
    }
The trace is the point: a ten-pattern system is only reviewable if each
pattern exposes *what it did* (sub-questions asked, chunks rejected,
graph hops taken), not just its final answer.
"""
from patterns import (
    adaptive,
    agentic,
    branched,
    corrective,
    graph_rag,
    hyde,
    memory_rag,
    multimodal,
    self_rag,
    simple,
)

PATTERNS = {
    "simple": simple.answer,
    "memory": memory_rag.answer,
    "branched": branched.answer,
    "hyde": hyde.answer,
    "adaptive": adaptive.answer,
    "corrective": corrective.answer,
    "self_rag": self_rag.answer,
    "agentic": agentic.answer,
    "multimodal": multimodal.answer,
    "graph": graph_rag.answer,
}

PATTERN_INFO = {
    "simple": "Single-pass retrieve → generate. The validated baseline (100% eval hit-rate).",
    "memory": "Conversation-aware: rewrites follow-up questions into standalone form using session history.",
    "branched": "Decomposes multi-part questions into sub-questions, retrieves for each, merges.",
    "hyde": "Writes a hypothetical statute passage first and searches with *its* embedding.",
    "adaptive": "Router: classifies the query and delegates to the best-suited pattern.",
    "corrective": "Grades retrieved chunks for relevance; rewrites the query and retries if they're bad.",
    "self_rag": "Drafts an answer, critiques it against the sources, regenerates if unsupported.",
    "agentic": "LLM drives a tool loop: search / fetch-section / follow-references / answer.",
    "multimodal": "Searches prose sections AND the structured-table index (rate schedules etc.).",
    "graph": "Expands retrieval one hop along real statutory cross-references (3,683 edges).",
}


def run(pattern_name, question, session_id=None):
    if pattern_name not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern_name!r}. Available: {sorted(PATTERNS)}")
    result = PATTERNS[pattern_name](question, session_id=session_id)
    result["pattern"] = result.get("pattern", pattern_name)
    return result
