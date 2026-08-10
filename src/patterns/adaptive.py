"""Pattern 5 - Adaptive RAG: classify the query, route to the right pattern.

The router is two-tier, cheapest check first:
  1. Deterministic: a query naming a section number routes straight to
     Simple RAG (whose exact-lookup path is guaranteed-correct) - no LLM
     call spent deciding what a regex already knows.
  2. LLM classification for everything else, one small JSON call:
       comparison  → branched   (needs both Acts in context)
       vague       → hyde       (colloquial phrasing far from statute prose)
       follow_up   → memory     (needs conversation history)
       rates       → multimodal (rate/threshold answers live in tables)
       topical     → simple     (the validated baseline)
       out_of_scope→ refuse immediately, zero retrieval cost

The inner pattern's full result (and trace) is returned, wrapped with the
routing decision - so the trace shows both *why* the route was chosen and
what the routed pattern did.
"""
import re

from generate import llm
from retrieve import _SECTION_QUERY_RE

_ROUTE_SYSTEM = """Classify a question for an Indian income-tax RAG system. Reply with JSON: {"category": "...", "reason": "..."}. Categories:
- "comparison": asks how the 1961 Act and 2025 Act differ, or mentions both/old-vs-new
- "vague": colloquial/situational phrasing far from legal language (e.g. "do I pay tax if my friend sends me money")
- "follow_up": clearly refers back to an earlier answer ("what about...", "and under the old act?", bare pronouns)
- "rates": asks for a tax rate, slab, threshold amount, or percentage
- "out_of_scope": not about Indian income tax law at all
- "topical": ordinary self-contained income-tax question (the default)"""

_OUT_OF_SCOPE_MESSAGE = (
    "This question is outside the corpus - I only cover the Income-tax Act, 2025 "
    "and the Income-tax Act, 1961."
)

_ROUTES = {
    "comparison": "branched",
    "vague": "hyde",
    "follow_up": "memory",
    "rates": "multimodal",
    "topical": "simple",
}


def answer(question, session_id=None):
    # avoid circular import: adaptive imports the registry lazily
    from patterns import PATTERNS

    if _SECTION_QUERY_RE.search(question):
        route, reason = "simple", "query names an explicit section number (deterministic fast path)"
    else:
        try:
            verdict = llm(f"Question: {question}", system_prompt=_ROUTE_SYSTEM, json_mode=True, max_tokens=120)
            category = str(verdict.get("category", "topical")).lower().strip()
            reason = verdict.get("reason", "")
        except ValueError:
            category, reason = "topical", "classifier failed, defaulting"
        if category == "out_of_scope":
            return {
                "answer": _OUT_OF_SCOPE_MESSAGE,
                "chunks": [],
                "refused": True,
                "trace": {"route": None, "category": "out_of_scope", "reason": reason},
            }
        route = _ROUTES.get(category, "simple")

    result = PATTERNS[route](question, session_id=session_id)
    inner_trace = result.get("trace", {})
    result["trace"] = {"route": route, "reason": reason, "inner": inner_trace}
    result["pattern"] = f"adaptive→{route}"
    return result
