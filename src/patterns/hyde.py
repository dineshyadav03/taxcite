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
from retrieve import _query_collection, exact_section_lookup, get_collection, vector_search

_HYDE_SYSTEM = """Write a short hypothetical passage (60-100 words) in the style of an Indian Income-tax Act section that would answer the user's question. Statutory register: "Every person...", "shall be chargeable...", "subject to the provisions of...". Do not cite real section numbers. Output only the passage."""


def answer(question, session_id=None):
    trace = {"hypothetical": None, "used_exact_path": False}

    exact = exact_section_lookup(question)
    if exact:
        trace["used_exact_path"] = True
        chunks = exact[:5]
    else:
        # Domain-relevance gate on the RAW question first, before ever
        # drafting a hypothetical. _HYDE_SYSTEM always writes in the
        # corpus's own statutory register ("shall be chargeable...",
        # "subject to the provisions of...") regardless of whether the
        # question has anything to do with tax law - verified live: a
        # chocolate-chip-cookie-recipe question produced a fake-but-
        # statute-shaped passage that embedded at distance 1.022,
        # comfortably under DISTANCE_REFUSAL_THRESHOLD, purely from
        # linguistic style rather than topic. That collapses the whole
        # "out of scope" signal, since any input becomes genre-matching
        # text before comparison. The raw question's own embedding
        # doesn't have this problem (it's what the threshold was
        # originally calibrated against) - verified live on both HyDE's
        # own founding example ("do I pay tax on money my friend
        # transferred me?", raw distance 0.936) and every golden-set
        # question HyDE previously got wrong (all measured well under
        # threshold on raw distance too, so this gate doesn't cost any of
        # HyDE's genuine value for awkwardly-phrased-but-on-topic
        # questions - it only catches genuinely off-corpus ones). Costs
        # one extra Voyage embedding call per non-exact HyDE query.
        raw_chunks = vector_search(question, top_k=1)
        trace["raw_distance"] = round(raw_chunks[0]["distance"], 3) if raw_chunks else None
        if not raw_chunks or raw_chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
            return {"answer": REFUSAL_MESSAGE, "chunks": raw_chunks, "refused": True, "trace": trace}

        # Raised from 200 after the Groq model swap (see generate.py's
        # GROQ_MODEL comment) - reasoning tokens count against this same
        # budget now, on top of the actual 60-100 word passage. 600 was
        # tried first and still measured too tight live (a real call
        # needed 611 tokens total and got cut off with empty output,
        # which then crashed the embedding call downstream on an empty
        # string) - 1000 leaves real headroom instead of sitting right at
        # the edge of a single measured sample.
        hypothetical = llm(question, system_prompt=_HYDE_SYSTEM, max_tokens=1000)
        trace["hypothetical"] = hypothetical
        hyde_embedding = embed_texts([hypothetical], input_type="document")[0]
        chunks = _query_collection(get_collection(), hyde_embedding, 5)

    if not chunks or chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True, "trace": trace}

    prompt = build_user_prompt(question, chunks)
    answer_text = llm(prompt)
    return {"answer": answer_text, "chunks": chunks, "refused": False, "trace": trace}
