"""Pattern 1 - Simple RAG: single-pass retrieve → grade → generate.

Originally a thin wrapper around generate.answer_question(); now does
its own retrieve+grade+generate so it can grade retrieved chunks (see
grading.py) before generating, while keeping generate.answer_question()
itself untouched - eval/run_eval.py's own baseline scoring counts on
that function's original, ungraded contract staying stable, so this
pattern no longer literally delegates to it, on purpose.

No retry on a bad grade - that stays Corrective RAG's differentiator. A
genuinely empty grade here is an honest refusal, the same outcome as if
nothing relevant had been retrieved at all pre-grading.
"""
from generate import DISTANCE_REFUSAL_THRESHOLD, REFUSAL_MESSAGE, build_user_prompt, llm
from grading import grade_chunks, grading_trace_entry
from retrieve import search


def answer(question, session_id=None, embedding=None):
    chunks = search(question, embedding=embedding)
    trace = {
        "retrieval": [
            {
                "section": c["metadata"].get("section"),
                "act": c["metadata"].get("act_id"),
                "distance": round(c["distance"], 3),
                "exact_match": c["distance"] == 0.0,
            }
            for c in chunks
        ]
    }

    if not chunks or chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    try:
        relevant, grades, _ = grade_chunks(question, chunks)
        trace["grading"] = grading_trace_entry(question, chunks, grades)
    except ValueError:
        relevant = chunks
        trace["grader_failed"] = True

    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, relevant)
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": relevant, "refused": False, "trace": trace}
