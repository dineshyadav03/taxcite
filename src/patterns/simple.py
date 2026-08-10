"""Pattern 1 - Simple RAG: single-pass retrieve → generate.

This is generate.answer_question with a trace wrapper. It stays the
baseline every other pattern is judged against: it scored 100% retrieval
hit-rate on the golden eval, so a fancier pattern that can't beat it on a
given query class is overhead, not progress.
"""
from generate import answer_question


def answer(question, session_id=None, embedding=None):
    result = answer_question(question, embedding=embedding)
    result["trace"] = {
        "retrieval": [
            {
                "section": c["metadata"].get("section"),
                "act": c["metadata"].get("act_id"),
                "distance": round(c["distance"], 3),
                "exact_match": c["distance"] == 0.0,
            }
            for c in result["chunks"]
        ]
    }
    return result
