"""Pattern 2 - RAG with memory: conversation-aware retrieval.

The mechanism that matters is query rewriting, not just prompt stuffing:
a follow-up like "and under the old Act?" embeds to garbage on its own -
no retrieval system can find "the thing we were just talking about, but
in the 1961 Act" from those five words. So when history exists, one LLM
call rewrites the follow-up into a standalone question first, and
*retrieval runs on the rewrite*. History also goes into the generation
prompt so the answer can acknowledge context.
"""
import memory
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
from grading import grade_chunks, grading_trace_entry
from retrieve import search

_REWRITE_SYSTEM = """You rewrite a follow-up question into a fully standalone question, using the conversation history to resolve pronouns and references. Reply with JSON: {"standalone_question": "..."}. If the question is already standalone, return it unchanged."""


def answer(question, session_id=None):
    session_id = session_id or "default"
    history = memory.format_history(session_id)
    trace = {"had_history": bool(history), "rewritten": None}

    retrieval_question = question
    if history:
        try:
            rewritten = llm(
                f"Conversation history:\n{history}\n\nFollow-up question: {question}",
                system_prompt=_REWRITE_SYSTEM,
                json_mode=True,
                max_tokens=150,
            )["standalone_question"]
            if rewritten and rewritten.strip() and rewritten.strip() != question.strip():
                retrieval_question = rewritten.strip()
                trace["rewritten"] = retrieval_question
        except (ValueError, KeyError):
            pass  # rewriting is an optimization; fall back to the raw question

    chunks = search(retrieval_question, top_k=5)
    if not chunks or chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        memory.add_turn(session_id, question, REFUSAL_MESSAGE)
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    try:
        relevant, grades, _ = grade_chunks(retrieval_question, chunks)
        trace["grading"] = grading_trace_entry(retrieval_question, chunks, grades)
    except ValueError:
        relevant = chunks
        trace["grader_failed"] = True

    if not relevant:
        memory.add_turn(session_id, question, REFUSAL_MESSAGE)
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(retrieval_question, relevant, history=history or None)
    answer_text = llm(prompt)
    memory.add_turn(session_id, question, answer_text)
    return {"answer": answer_text, "chunks": relevant, "refused": False, "trace": trace}
