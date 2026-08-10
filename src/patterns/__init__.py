"""Registry of the ten standard RAG patterns plus three original ones
(jury, correspondence, precedent) built on the same substrate.

Every pattern is a callable answer(question, session_id=None) returning:
    {
      "answer":  str,
      "chunks":  [retrieved chunk dicts],
      "refused": bool,
      "trace":   {pattern-specific record of what actually happened},
      "pattern": str,
    }
The trace is the point: a thirteen-pattern system is only reviewable if
each pattern exposes *what it did* (sub-questions asked, chunks
rejected, graph hops taken), not just its final answer.
"""
from patterns import (
    adaptive,
    agentic,
    branched,
    correspondence,
    corrective,
    graph_rag,
    hyde,
    jury,
    memory_rag,
    multimodal,
    precedent,
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
    "jury": jury.answer,
    "correspondence": correspondence.answer,
    "precedent": precedent.answer,
}

# best_for/weak_for/speed are not design intent - they're the result of
# actually asking all 13 patterns the identical question in one sitting
# and recording what happened (see README.md's "Which pattern should you
# use?" for the full test and reasoning). speed labels are the real
# observed order-of-magnitude, not a guess: fast = well under a couple
# seconds, medium = a few seconds, slow = 60s+ (verified live: Branched
# 87s, Agentic 72s, Jury ~100-175s - all patterns that do more than one
# retrieval pass under Voyage's free-tier 3-requests/minute throttle).
PATTERN_INFO = {
    "simple": {
        "description": "Single-pass retrieve → generate. The validated baseline (100% eval hit-rate).",
        "best_for": "An exact section-number question, or a well-phrased topical question with one clear answer.",
        "weak_for": "Broad or comparative questions with no single matching section - refuses rather than guess.",
        "speed": "fast",
    },
    "memory": {
        "description": "Conversation-aware: rewrites follow-up questions into standalone form using session history.",
        "best_for": "Multi-turn conversations - \"what about the old Act?\" after an earlier answer.",
        "weak_for": "Standalone questions; shares Simple's weakness on broad questions since it's the same retrieval underneath.",
        "speed": "fast",
    },
    "branched": {
        "description": "Decomposes multi-part questions into sub-questions, retrieves for each, merges.",
        "best_for": "Explicit multi-part or \"compare old vs. new\" questions - the pattern actually built for them.",
        "weak_for": "Simple single-section lookups, where decomposition is pure overhead.",
        "speed": "slow",
    },
    "hyde": {
        "description": "Writes a hypothetical statute passage first and searches with *its* embedding.",
        "best_for": "Vague or colloquial phrasing far from statute language.",
        "weak_for": "Nothing observed in testing - handled a broad comparative question every other single-pass pattern refused.",
        "speed": "medium",
    },
    "adaptive": {
        "description": "Router: classifies the query and delegates to the best-suited pattern.",
        "best_for": "The default choice when you don't know which pattern fits - routes automatically.",
        "weak_for": "Nothing specific - only as good as whichever pattern it routes to.",
        "speed": "fast (then inherits the routed pattern's time)",
    },
    "corrective": {
        "description": "Grades retrieved chunks for relevance; rewrites the query and retries if they're bad.",
        "best_for": "A general-purpose upgrade over Simple - handled the same broad question that stumped Simple/Memory/Graph.",
        "weak_for": "Nothing specific observed.",
        "speed": "medium",
    },
    "self_rag": {
        "description": "Drafts an answer, critiques it against the sources, regenerates if unsupported.",
        "best_for": "When you want the system to double-check its own answer against the source before returning it.",
        "weak_for": "Shares Simple's single-shot retrieval weakness - the critique pass helps faithfulness, not retrieval.",
        "speed": "medium",
    },
    "agentic": {
        "description": "LLM drives a tool loop: search / fetch-section / follow-references / answer.",
        "best_for": "Multi-hop questions needing several fetch/follow-reference steps chosen by the model itself.",
        "weak_for": "Simple lookups, where the tool loop is pure overhead.",
        "speed": "slow",
    },
    "multimodal": {
        "description": "Searches prose sections AND the structured-table index (rate schedules etc.).",
        "best_for": "Rate, slab, and threshold questions - anything living in a Schedule table rather than prose.",
        "weak_for": "Broad comparative questions - refuses the same way Simple does.",
        "speed": "fast (occasionally slower if a table-search retry triggers)",
    },
    "graph": {
        "description": "Expands retrieval one hop along real statutory cross-references (3,683 edges).",
        "best_for": "A question about a section's own dependencies - what it cites, what depends on it.",
        "weak_for": "Shares Simple's single-shot retrieval weakness on broad questions.",
        "speed": "fast",
    },
    "jury": {
        "description": "Runs simple/corrective/graph as an ensemble; reports consensus or explicit disagreement.",
        "best_for": "A single-answer question where you want a confidence signal - real agreement between 3 independent methods.",
        "weak_for": "Broad comparative questions - none of its 3 jurors are built for whole-Act comparison, so the vote is confusing.",
        "speed": "slow",
    },
    "correspondence": {
        "description": "Maps a 1961-Act section to its verified 2025-Act functional equivalent.",
        "best_for": "Exactly one job: \"what's the 2025-Act equivalent of 1961 Section N?\"",
        "weak_for": "Everything else - explicitly declines rather than answer a question it isn't built for.",
        "speed": "fast (no embedding call at all)",
    },
    "precedent": {
        "description": "Attaches real, verified Supreme Court case law to the retrieved statute section.",
        "best_for": "A question about one of the 18 curated sections with real linked case law.",
        "weak_for": "Any section outside that curated set; shares Simple's single-shot weakness otherwise.",
        "speed": "fast",
    },
}


def run(pattern_name, question, session_id=None):
    if pattern_name not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern_name!r}. Available: {sorted(PATTERNS)}")
    result = PATTERNS[pattern_name](question, session_id=session_id)
    result["pattern"] = result.get("pattern", pattern_name)
    return result
