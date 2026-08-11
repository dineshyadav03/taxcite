"""BM25 keyword search over the section corpus - the other half of hybrid
retrieval (see retrieve.search()'s RRF merge).

Vector similarity is the system's main retrieval mechanism, but it has a
real, documented gap: a topical query with no section number relies on
embedding similarity alone, which can miss the best match the way any
pure similarity search can - a query using an exact, lexically
distinctive statutory term doesn't always score close to the section
that contains that literal term, if the surrounding phrasing differs.
BM25 is the classical fix for exactly this: it scores on literal term
overlap (with length-normalization and term-frequency saturation), which
vector similarity doesn't directly capture.

Built once from the already-indexed corpus (zero new API calls - the
text is already in ChromaDB from the original embedding build) and kept
in memory, same "rebuild fresh on process start" choice as the
cross-reference graph: at ~2,400 chunks this takes well under a second,
so persisting an index to disk would be pure overhead.
"""
import re

from rank_bm25 import BM25Okapi

from retrieve import get_collection

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25 = None
_ids = None
_documents = None
_metadatas = None


def _tokenize(text):
    return _TOKEN_RE.findall((text or "").lower())


def _build():
    global _bm25, _ids, _documents, _metadatas
    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    _ids = result["ids"]
    _documents = result["documents"]
    _metadatas = result["metadatas"]
    tokenized = [_tokenize(doc) for doc in _documents]
    _bm25 = BM25Okapi(tokenized)


# Not a real distance - BM25 scores aren't comparable to cosine distance,
# only to other BM25 scores, and retrieve.search()'s RRF merge combines
# rankers by rank position, not by this value. Exists purely so a
# BM25-only chunk (one vector search didn't also find) doesn't crash
# every pattern's trace-building code, which does round(c["distance"], 3)
# on every retrieved chunk, not just the position-0 one the refusal gate
# checks - verified live: a bare `None` here crashed graph_rag.py's trace
# with "TypeError: type NoneType doesn't define __round__ method" the
# first time a BM25-only chunk landed past position 0. Never read by the
# refusal gate itself (search() guarantees chunks[0] always carries a
# real vector distance), so this number is cosmetic, not load-bearing.
_SENTINEL_DISTANCE = 1.0


def bm25_search(query, top_k=10):
    """Keyword-ranked chunks for query, same chunk-dict shape other
    retrieval functions return ({id, text, metadata, distance}) so callers
    can merge results uninterested in which ranker produced them."""
    if _bm25 is None:
        _build()
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"id": _ids[i], "text": _documents[i], "metadata": _metadatas[i], "distance": _SENTINEL_DISTANCE}
        for i in ranked
        if scores[i] > 0
    ]
