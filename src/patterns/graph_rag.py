"""Pattern 10 - Graph RAG: expand retrieval along statutory cross-references.

The gap this fills: vector similarity can't see that understanding
Section 172 *requires* Section 511 - the texts aren't semantically alike,
they're just legally coupled ("as prescribed under section 511(1)"). The
cross-reference graph (src/graph.py: 3,683 edges regex-extracted from the
section texts themselves) makes that coupling retrievable: after normal
search, the top hits' outgoing references are fetched too, so definitions
and procedural anchors arrive in context alongside the section that
depends on them.

Lightweight by design - a real property graph DB would add operational
weight without changing the mechanism at this corpus size. One hop only:
the reference closure of a tax statute is dense enough that two hops
would pull half the Act into context.
"""
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
import graph
from retrieve import fetch_section, search

_SEED_K = 4       # vector/exact hits to seed from
_EXPAND_TOP = 2   # seeds whose references get followed
_REFS_PER_SEED = 3
_MAX_CONTEXT = 8


def answer(question, session_id=None):
    trace = {"seeds": [], "expansions": {}}
    seeds = search(question, top_k=_SEED_K)

    if not seeds or seeds[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": seeds, "refused": True, "trace": trace}

    trace["seeds"] = [
        {"section": c["metadata"].get("section"), "act": c["metadata"].get("act_id"), "distance": round(c["distance"], 3)}
        for c in seeds
    ]

    merged = list(seeds)
    seen_ids = {c["id"] for c in merged}
    seen_sections = {(c["metadata"]["act_id"], c["metadata"]["section"]) for c in merged}

    for seed in seeds[:_EXPAND_TOP]:
        act_id = seed["metadata"]["act_id"]
        section = seed["metadata"]["section"]
        refs = graph.references_of(act_id, section)[:_REFS_PER_SEED]
        followed = []
        for ref in refs:
            if (act_id, ref) in seen_sections:
                continue
            ref_chunks = fetch_section(act_id, ref)[:1]  # first chunk carries the section's charge/definition
            for c in ref_chunks:
                if c["id"] not in seen_ids:
                    c["via_reference"] = f"{section} → {ref}"
                    merged.append(c)
                    seen_ids.add(c["id"])
                    seen_sections.add((act_id, ref))
                    followed.append(ref)
        if followed:
            trace["expansions"][f"{act_id}::{section}"] = followed

    prompt = build_user_prompt(question, merged[:_MAX_CONTEXT])
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": merged[:_MAX_CONTEXT], "refused": False, "trace": trace}
