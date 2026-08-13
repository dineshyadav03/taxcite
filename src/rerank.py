"""Reranking over hybrid retrieval's fused candidates - the shared final
step in retrieve.search() before the exact-match-anchored top slot.

Vector similarity and BM25 (see bm25_index.py) each retrieve candidates
by a cheap, approximate signal - RRF fusion combines their rankings, but
neither ranker re-reads a candidate against the query with a model built
specifically to judge relevance. Reranking is that step: Voyage's hosted
rerank-2-lite scores each fused candidate directly against the query,
free-tier covered under the same 200M-token pool as embeddings (verified
via Voyage's own pricing page before choosing this over reintroducing a
local cross-encoder, which would have undone the earlier move off local
ML dependencies entirely).

Real, stated cost: this is a second Voyage API call per query on top of
the embedding call, under the same 3-requests/minute free-tier ceiling
that already caused several real rate-limit bugs this session. Applied
everywhere by design (see the project's own production-checklist plan),
not because the cost is free - it isn't.
"""
from embed import get_client

RERANK_MODEL = "rerank-2-lite"


def rerank(query, chunks, top_k):
    """Re-scores chunks against query with Voyage's hosted reranker,
    returns the top_k in relevance order. chunks keep their original
    dicts (id/text/metadata/distance) - only the ORDER and SELECTION
    change, distance values are untouched (a reranked chunk's distance
    still reflects whichever ranker originally found it, same as
    retrieve.search()'s existing RRF merge already does)."""
    if not chunks:
        return chunks
    client = get_client()
    result = client.rerank(
        query=query,
        documents=[c["text"] for c in chunks],
        model=RERANK_MODEL,
        top_k=min(top_k, len(chunks)),
    )
    ranked = sorted(result.results, key=lambda r: r.relevance_score, reverse=True)
    return [chunks[r.index] for r in ranked]
