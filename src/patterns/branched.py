"""Pattern 3 - Branched RAG: decompose, retrieve per branch, merge.

Built for questions a single retrieval pass structurally can't serve:
"how does section 139 differ between the old and new Act" needs *both*
Acts' sections in context, but one vector query returns whichever scores
higher. Decomposition turns it into per-Act sub-questions, each of which
retrieves independently (and each sub-question naming a section number
gets the exact-lookup fast path for free, since retrieval is shared).
"""
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
from retrieve import search

_DECOMPOSE_SYSTEM = """You decompose a question about Indian income tax law into 2-3 focused sub-questions that can each be answered from a single statute lookup. Reply with JSON: {"sub_questions": ["...", "..."]}. If the question is already atomic, return it as the only sub-question. When the question compares the Income-tax Act 1961 and the Income-tax Act 2025, make one sub-question per Act, each explicitly naming its Act."""

_PER_BRANCH_K = 3
_MAX_BRANCHES = 3


def answer(question, session_id=None):
    trace = {"sub_questions": [], "per_branch_hits": []}
    try:
        subs = llm(
            f"Question: {question}",
            system_prompt=_DECOMPOSE_SYSTEM,
            json_mode=True,
            max_tokens=200,
        )["sub_questions"][:_MAX_BRANCHES]
    except (ValueError, KeyError):
        subs = [question]
    if not subs:
        subs = [question]
    trace["sub_questions"] = subs

    merged, seen = [], set()
    for sub in subs:
        hits = search(sub, top_k=_PER_BRANCH_K)
        trace["per_branch_hits"].append(
            [
                {"section": c["metadata"].get("section"), "act": c["metadata"].get("act_id"), "distance": round(c["distance"], 3)}
                for c in hits
            ]
        )
        for c in hits:
            if c["id"] not in seen:
                merged.append(c)
                seen.add(c["id"])

    relevant = [c for c in merged if c["distance"] <= DISTANCE_REFUSAL_THRESHOLD]
    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "chunks": merged, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, relevant[:8])
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": relevant[:8], "refused": False, "trace": trace}
