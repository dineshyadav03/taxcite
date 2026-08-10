"""Pattern 5(6) - Corrective RAG: grade retrieval, retry if it's bad.

The failure this catches: vector search always returns *something* - its
top-k is "the least-distant chunks", not "relevant chunks". One grading
call classifies every retrieved chunk as relevant/irrelevant to the
actual question. If too few survive, the query is rewritten (the grader's
feedback becomes the rewrite hint) and retrieval runs once more. Only
graded-relevant chunks ever reach the generator, so an off-target chunk
can't quietly steer the answer.

One retry, not a loop: each retry costs an embedding call (3 RPM budget)
and a grading call, and if two well-formed retrievals both fail grading,
the honest output is a refusal, not a third attempt.
"""
from generate import REFUSAL_MESSAGE, build_user_prompt, llm
from retrieve import search

_GRADE_SYSTEM = """You grade retrieved statute excerpts for relevance to a question. For each excerpt decide: does it contain information that helps answer this specific question? Reply with JSON: {"grades": [{"index": 0, "relevant": true/false, "reason": "..."}], "better_query": "an improved search query if most excerpts are irrelevant, else null"}."""

_MIN_RELEVANT = 2


def _grade(question, chunks):
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('act_title')}, Section {c['metadata'].get('section')})\n{c['text'][:500]}"
        for i, c in enumerate(chunks)
    )
    verdict = llm(
        f"Question: {question}\n\nExcerpts:\n{listing}",
        system_prompt=_GRADE_SYSTEM,
        json_mode=True,
        max_tokens=400,
    )
    grades = {g["index"]: g for g in verdict.get("grades", []) if isinstance(g, dict) and "index" in g}
    relevant = [c for i, c in enumerate(chunks) if grades.get(i, {}).get("relevant")]
    return relevant, grades, verdict.get("better_query")


def answer(question, session_id=None):
    trace = {"rounds": []}
    chunks = search(question, top_k=6)

    try:
        relevant, grades, better_query = _grade(question, chunks)
        trace["rounds"].append(
            {
                "query": question,
                "graded": [
                    {"section": c["metadata"].get("section"), "act": c["metadata"].get("act_id"), "relevant": grades.get(i, {}).get("relevant", False)}
                    for i, c in enumerate(chunks)
                ],
            }
        )
    except ValueError:
        # grader output unparseable - degrade to Simple RAG behavior
        relevant, better_query = chunks, None
        trace["grader_failed"] = True

    if len(relevant) < _MIN_RELEVANT and better_query:
        trace["rewritten_query"] = better_query
        retry_chunks = search(better_query, top_k=6)
        try:
            retry_relevant, retry_grades, _ = _grade(question, retry_chunks)
            trace["rounds"].append(
                {
                    "query": better_query,
                    "graded": [
                        {"section": c["metadata"].get("section"), "act": c["metadata"].get("act_id"), "relevant": retry_grades.get(i, {}).get("relevant", False)}
                        for i, c in enumerate(retry_chunks)
                    ],
                }
            )
        except ValueError:
            retry_relevant = []
        seen = {c["id"] for c in relevant}
        relevant += [c for c in retry_relevant if c["id"] not in seen]

    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, relevant[:6])
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": relevant[:6], "refused": False, "trace": trace}
