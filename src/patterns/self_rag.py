"""Pattern 7 - Self-RAG: draft, self-critique against sources, revise.

Where Corrective RAG guards the *input* (are these chunks relevant?),
Self-RAG guards the *output*: after drafting, a second call checks every
claim in the draft against the retrieved text and lists unsupported ones.
If any exist, one regeneration runs with the critique as an explicit
constraint. For a legal corpus this is the highest-value check there is -
the failure mode that matters most is a fluent answer citing a section
that doesn't say what the answer claims.

The critique is returned in the trace even when the draft passes, so a
reviewer can see the check actually ran.
"""
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
from retrieve import search

_CRITIQUE_SYSTEM = """You verify a drafted answer against the statute excerpts it was based on. Check every factual claim and citation. Reply with JSON: {"fully_supported": true/false, "unsupported_claims": ["quote each claim not supported by the excerpts"], "verdict": "one-sentence overall judgment"}."""


def answer(question, session_id=None):
    trace = {"critique": None, "revised": False}
    chunks = search(question, top_k=5)

    if not chunks or chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, chunks)
    draft = llm(prompt)

    context = "\n\n".join(
        f"({c['metadata'].get('act_title')}, Section {c['metadata'].get('section')})\n{c['text'][:600]}"
        for c in chunks
    )
    try:
        critique = llm(
            f"Excerpts:\n{context}\n\nDrafted answer:\n{draft}",
            system_prompt=_CRITIQUE_SYSTEM,
            json_mode=True,
            # Raised from 600 after the Groq model swap (see generate.py's
            # GROQ_MODEL comment) - reasoning tokens count against this
            # same budget now.
            max_tokens=1500,
        )
        trace["critique"] = critique
    except Exception as e:
        # Groq's JSON-mode validator can hard-error (400,
        # json_validate_failed) rather than returning truncated text to
        # salvage - verified live, a plain ValueError catch didn't cover
        # it. The critique is a quality check, not the answer itself, so
        # degrade to "draft stands as-is" rather than failing the whole
        # pattern over a verification-step hiccup.
        critique = None
        trace["critique"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}

    answer_text = draft
    if critique and not critique.get("fully_supported", True):
        unsupported = critique.get("unsupported_claims", [])
        revision_prompt = (
            prompt
            + "\n\nA verification pass found these claims in a previous draft were NOT supported by the excerpts:\n- "
            + "\n- ".join(str(u) for u in unsupported)
            + "\n\nWrite a corrected answer that omits or fixes the unsupported claims. Only state what the excerpts support."
        )
        answer_text = llm(revision_prompt)
        trace["revised"] = True

    return {"answer": answer_text, "chunks": chunks, "refused": False, "trace": trace}
