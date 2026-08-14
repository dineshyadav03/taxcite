"""Pattern 9 - Multi-modal RAG: prose sections + structured tables.

Honest framing: a statute corpus has no images or audio, so "multi-modal"
here means the two genuinely different representations this corpus does
have - prose section text, and the rate-schedule/Schedule tables that the
prose pipeline extracts as garbled noise (a documented failure: the 2025
Act's Section 194 rate table caused degenerate LLM repetition). Tables
live in their own collection, extracted structurally (pdfplumber cell
grids serialized as pipe-delimited rows) instead of as interleaved prose.

One embedding call serves both collections - the query vector is computed
once and reused, which matters at 3 requests/minute.
"""
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
from grading import grade_chunks, grading_trace_entry
from retrieve import embed_query, exact_section_lookup, table_search, vector_search

_PROSE_K = 4
_TABLE_K = 3


def answer(question, session_id=None):
    trace = {"prose_hits": [], "table_hits": []}

    exact = exact_section_lookup(question)
    embedding = embed_query(question)
    prose = vector_search(question, top_k=_PROSE_K, embedding=embedding)
    tables = table_search(question, top_k=_TABLE_K, embedding=embedding)

    trace["prose_hits"] = [
        {"section": c["metadata"].get("section"), "act": c["metadata"].get("act_id"), "distance": round(c["distance"], 3)}
        for c in prose
    ]
    trace["table_hits"] = [
        {"section": c["metadata"].get("section") or "?", "act": c["metadata"].get("act_id"), "page": c["metadata"].get("page"), "distance": round(c["distance"], 3)}
        for c in tables
    ]

    # Prose is held to the same confidence bar every other pattern uses.
    # Tables get a wider one: serialized pipe-delimited rows sit farther
    # from a query embedding than prose describing the same provision,
    # even when genuinely relevant, so the standard 1.15 under-retrieves
    # real table matches. That buffer must stay scoped to tables only -
    # verified live: applying it to prose too let an off-corpus question
    # ("GST rate on restaurant services", distance 1.223) slip past the
    # gate on its prose hit, which the grader then correctly rejected,
    # but the borderline TABLE hit for the same section (distance 1.251,
    # also only passable under the loosened bar) got graded relevant -
    # a bare numeric rate table has little context for the grader to
    # judge relevance from, so the wider threshold needs prose's stricter
    # bar to stay in place as a first line of defense, not be loosened
    # right along with it.
    table_hits = [t for t in tables if t["distance"] <= DISTANCE_REFUSAL_THRESHOLD + 0.15]
    prose_hits_relevant = [p for p in prose if p["distance"] <= DISTANCE_REFUSAL_THRESHOLD]

    merged, seen = [], set()
    for c in exact + prose_hits_relevant + table_hits:
        if c["id"] not in seen:
            merged.append(c)
            seen.add(c["id"])

    if not merged:
        return {"answer": REFUSAL_MESSAGE, "chunks": exact + prose + tables, "refused": True, "trace": trace}

    candidates = merged[:7]
    try:
        graded, grades, _ = grade_chunks(question, candidates)
        trace["grading"] = grading_trace_entry(question, candidates, grades)
    except ValueError:
        graded = candidates
        trace["grader_failed"] = True

    if not graded:
        return {"answer": REFUSAL_MESSAGE, "chunks": candidates, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, graded)
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": graded, "refused": False, "trace": trace}
