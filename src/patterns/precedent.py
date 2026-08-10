"""Pattern 13 - Precedent RAG: attach real case law to a statute answer.

Every other pattern answers purely from statute text. This one also
surfaces the Supreme Court judgments that actually interpret the
retrieved section, via graph.cases_for() - built from
data/processed/case_graph.json, 18 real cases sourced from a verified
digest and manually linked to sections (see scripts/build_cases.py).

Same DISTANCE_REFUSAL_THRESHOLD gate as every other pattern: a poor
statute match refuses before ever attempting to attach a case, since a
case attached to the wrong section would compound one bad guess with
another. If the matched section has no linked case (true for the vast
majority of sections - only 16 of them are in the curated case set),
the answer says so plainly rather than silently returning a statute-only
answer with no explanation of why no case appeared.
"""
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
import graph
from retrieve import get_cases_collection, search

_SEED_K = 4

_PRECEDENT_SYSTEM = """You are a research assistant answering questions about Indian income tax law, grounded ONLY in the retrieved Act sections and any linked case law provided below. Rules:
- Answer the statute question using only the retrieved text, citing the real Act title and section number.
- If case law is provided, explain how the case's holding bears on the question - but do not treat the case's holding as itself part of the statute text; keep the statutory rule and the judicial interpretation of it clearly distinct.
- If no case law is provided, answer from the statute alone.
- Do not use outside knowledge of tax law or of any case not provided to you."""

_NO_PRECEDENT_NOTE = (
    "No linked precedent for this section in the current case set (18 curated Supreme "
    "Court judgments) - this is a statute-only answer."
)


def answer(question, session_id=None):
    trace = {"seeds": [], "linked_cases": []}
    seeds = search(question, top_k=_SEED_K)

    if not seeds or seeds[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": seeds, "refused": True, "trace": trace}

    trace["seeds"] = [
        {"section": c["metadata"].get("section"), "act": c["metadata"].get("act_id"), "distance": round(c["distance"], 3)}
        for c in seeds
    ]

    seen_sections = set()
    linked = []
    for c in seeds:
        key = (c["metadata"]["act_id"], c["metadata"]["section"])
        if key in seen_sections:
            continue
        seen_sections.add(key)
        for case_name in graph.cases_for(*key):
            if case_name not in [x["case_name"] for x in linked]:
                linked.append({"case_name": case_name, "act": key[0], "section": key[1]})

    if not linked:
        prompt = build_user_prompt(question, seeds)
        answer_text = llm(prompt, system_prompt=_PRECEDENT_SYSTEM)
        answer_text = f"{answer_text}\n\n{_NO_PRECEDENT_NOTE}"
        trace["linked_cases"] = []
        return {"answer": answer_text, "chunks": seeds, "refused": False, "trace": trace}

    cases_collection_hits = _fetch_case_details([x["case_name"] for x in linked])
    trace["linked_cases"] = cases_collection_hits

    case_listing = "\n\n".join(
        f"Case: {c['case_name']} ({c['citation']})\nHolding: {c['holding']}" for c in cases_collection_hits
    )
    prompt = build_user_prompt(question, seeds) + f"\n\n---\n\nLinked case law:\n{case_listing}"
    answer_text = llm(prompt, system_prompt=_PRECEDENT_SYSTEM)

    return {"answer": answer_text, "chunks": seeds, "refused": False, "trace": trace}


def _fetch_case_details(case_names):
    all_metas = get_cases_collection().get(include=["metadatas"])["metadatas"]
    by_name = {m["case_name"]: m for m in all_metas}
    return [by_name[name] for name in case_names if name in by_name]
