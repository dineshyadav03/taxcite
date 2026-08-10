"""Pattern 4 - HyDE (Hypothetical Document Embeddings).

The asymmetry HyDE exploits: a colloquial question ("do I pay tax on
money my friend transferred me?") embeds far from statute prose, but a
*hypothetical statute passage answering it* embeds close to the real one.
So: LLM drafts a fake ~80-word statutory passage, that draft is embedded
with input_type="document" (it reads like a document - that's the whole
trick), and the index is searched with the draft's vector instead of the
question's.

A query that names a section number skips HyDE entirely: the exact-lookup
path is already guaranteed-correct, and a hypothetical would only add an
embedding call (3 RPM budget) for nothing.
"""
from embed import embed_texts
from generate import (
    DISTANCE_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    build_user_prompt,
    llm,
)
from retrieve import _query_collection, exact_section_lookup, get_collection

_HYDE_SYSTEM = """Write a short hypothetical passage (60-100 words) in the style of an Indian Income-tax Act section that would answer the user's question. Statutory register: "Every person...", "shall be chargeable...", "subject to the provisions of...". Do not cite real section numbers. Output only the passage."""


def answer(question, session_id=None):
    trace = {"hypothetical": None, "used_exact_path": False}

    exact = exact_section_lookup(question)
    if exact:
        trace["used_exact_path"] = True
        chunks = exact[:5]
    else:
        hypothetical = llm(question, system_prompt=_HYDE_SYSTEM, max_tokens=200)
        trace["hypothetical"] = hypothetical
        hyde_embedding = embed_texts([hypothetical], input_type="document")[0]
        chunks = _query_collection(get_collection(), hyde_embedding, 5)

    if not chunks or chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, chunks)
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": chunks, "refused": False, "trace": trace}
