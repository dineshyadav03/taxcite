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
from grading import grade_chunks, grading_trace_entry
from retrieve import search

_MIN_RELEVANT = 2


def answer(question, session_id=None, embedding=None):
    trace = {"rounds": []}
    chunks = search(question, top_k=6, embedding=embedding)

    try:
        relevant, grades, better_query = grade_chunks(question, chunks)
        trace["rounds"].append(grading_trace_entry(question, chunks, grades))
    except ValueError:
        # grader output unparseable - degrade to Simple RAG behavior
        relevant, better_query = chunks, None
        trace["grader_failed"] = True

    if len(relevant) < _MIN_RELEVANT and better_query:
        trace["rewritten_query"] = better_query
        retry_chunks = search(better_query, top_k=6)
        try:
            retry_relevant, retry_grades, _ = grade_chunks(question, retry_chunks)
            trace["rounds"].append(grading_trace_entry(question, retry_chunks, retry_grades))
        except ValueError:
            retry_relevant = []
        seen = {c["id"] for c in relevant}
        relevant += [c for c in retry_relevant if c["id"] not in seen]

    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, relevant[:6])
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": relevant[:6], "refused": False, "trace": trace}
