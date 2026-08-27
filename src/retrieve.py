"""Vector retrieval over the income-tax section index.

A query like "what does section 194 say" is semantically generic (it
names a number, not a topic), so a general-purpose embedding model can't
reliably match it to the right section - verified directly: the true
Section 194 of the 2025 Act sat at distance 1.18, while an unrelated 1961
Act section that merely *mentions* "194" in a list of cross-references
scored 0.61 and won. Exact section citations are the entire point of this
project, so this gets a direct regex-triggered metadata-filter lookup
rather than waiting on a future hybrid-search phase.
"""
import re
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import chromadb

from embed import embed_texts

ROOT = Path(__file__).resolve().parent.parent

_collections = {}

_SECTION_QUERY_RE = re.compile(r"\bsection\s+(\d{1,4}[A-Z]{0,4}(?:-[A-Z0-9]{1,4})?)\b", re.IGNORECASE)
# Order-insensitive on purpose - "the 2025 Act" and "Income-tax Act, 2025"
# both occur naturally, and the original single-direction regex
# (r"\b(2025|new)\s+act\b") missed the second form entirely, verified
# live: "What does section 139 of the Income-tax Act 2025 say?" silently
# resolved to no act_filter.
_ACT_2025_RE = re.compile(r"\b(?:(?:2025|new)\s+act|act,?\s+2025)\b", re.IGNORECASE)
_ACT_1961_RE = re.compile(r"\b(?:(?:1961|old)\s+act|act,?\s+1961)\b", re.IGNORECASE)


def _get(name, persist_dir=None):
    if name not in _collections:
        persist_dir = str(persist_dir or ROOT / "chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        _collections[name] = client.get_collection(name)
    return _collections[name]


def get_collection(persist_dir=None):
    return _get("income_tax_sections", persist_dir)


def get_table_collection(persist_dir=None):
    return _get("income_tax_tables", persist_dir)


def get_cases_collection(persist_dir=None):
    return _get("income_tax_cases", persist_dir)


def embed_queries(queries):
    """Batch-embed several distinct query texts in ONE Voyage call instead
    of one call per query - the free tier throttles by request count (3
    RPM), not by text count, so batching several genuinely-different
    queries (e.g. Branched RAG's per-Act sub-questions) into one call
    matters exactly as much as reusing one embedding across identical
    queries does (see jury.py). Order-preserving: result[i] is queries[i]'s
    embedding."""
    return embed_texts(list(queries), input_type="query")


def embed_query(query):
    """One Voyage call for a query embedding, reusable across collections.
    Patterns that search multiple collections (multi-modal) or reuse an
    embedding (HyDE) should embed once and pass the vector around - the
    free tier is 3 requests/minute, so every avoided call matters."""
    # input_type="query" vs. the index's "document" - Voyage embeds the two
    # asymmetrically for better retrieval quality, so this must match what
    # build_index() used or the vector space comparison is inconsistent.
    return embed_texts([query], input_type="query")[0]


def _query_collection(collection, query_embedding, top_k):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return chunks


def vector_search(query, top_k=5, embedding=None):
    if embedding is None:
        embedding = embed_query(query)
    return _query_collection(get_collection(), embedding, top_k)


def table_search(query, top_k=3, embedding=None):
    """Vector search over the table-modality collection. Returns [] if the
    table index hasn't been built yet rather than crashing."""
    if embedding is None:
        embedding = embed_query(query)
    try:
        return _query_collection(get_table_collection(), embedding, top_k)
    except Exception:
        return []


def fetch_section(act_id, section):
    """All chunks of one exact section, in order - direct metadata fetch,
    no embedding call. Used by the agentic tools and graph expansion."""
    collection = get_collection()
    results = collection.get(
        where={"$and": [{"section": section}, {"act_id": act_id}]},
        include=["documents", "metadatas"],
    )
    chunks = [
        {"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i], "distance": 0.0}
        for i in range(len(results["ids"]))
    ]
    chunks.sort(key=lambda c: c["metadata"]["chunk_index"])
    return chunks


def exact_section_lookup(query):
    """If the query names a section number, fetch it directly by metadata
    filter (distance=0.0, i.e. a guaranteed exact match) rather than
    relying on embedding similarity. Scoped to one Act if the query names
    one ("the 2025 Act" / "the old Act"), else both."""
    m = _SECTION_QUERY_RE.search(query)
    if not m:
        return []
    section_no = m.group(1)

    # Independent checks, not if/elif - a comparison question naturally
    # names BOTH Acts ("the old 1961 Act and the new 2025 Act"), and the
    # original elif matched 2025 first, silently dropping the 1961 side
    # entirely for every such question. Verified live: this was still
    # quietly breaking compare-01 (the project's own flagship 1961-vs-
    # 2025 example) - a grading-layer fix earlier this session made a
    # single-Act partial match survive grading, but the ROOT gap (the
    # 1961 Act's exact match never being fetched at all) was never
    # actually closed, and a more conservative model surfaced it again
    # by honestly declining to compare with only one side in hand rather
    # than answering from incomplete context anyway. Now: filter to one
    # Act only when the query unambiguously names just that one; a query
    # naming both (or neither) searches both, matching this function's
    # own docstring, not just its previous behavior.
    mentions_2025 = bool(_ACT_2025_RE.search(query))
    mentions_1961 = bool(_ACT_1961_RE.search(query))
    act_filter = None
    if mentions_2025 and not mentions_1961:
        act_filter = "itact2025"
    elif mentions_1961 and not mentions_2025:
        act_filter = "itact1961"

    where = {"section": section_no}
    if act_filter:
        where = {"$and": [{"section": section_no}, {"act_id": act_filter}]}

    collection = get_collection()
    results = collection.get(where=where, include=["documents", "metadatas"])
    chunks = [
        {"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i], "distance": 0.0}
        for i in range(len(results["ids"]))
    ]

    # Interleave by act, not a flat sort-then-slice: the same section
    # number can exist in both Acts with very different lengths (e.g.
    # Section 139 runs to 40k+ chars in the 1961 Act but is short in the
    # 2025 Act), and a plain sort-by-(act_id, chunk_index) lets whichever
    # act sorts first eat the entire result before callers ever slice to
    # top_k - verified live: "section 139 of the 2025 Act" returned only
    # 1961-Act chunks because itact1961 < itact2025 alphabetically and
    # itact1961's section 139 alone has more chunks than most top_k values.
    by_act = defaultdict(list)
    for c in chunks:
        by_act[c["metadata"]["act_id"]].append(c)
    for act_chunks in by_act.values():
        act_chunks.sort(key=lambda c: c["metadata"]["chunk_index"])

    interleaved = []
    for round_chunks in zip_longest(*by_act.values()):
        interleaved.extend(c for c in round_chunks if c is not None)
    return interleaved


def _rrf_merge(rankings, k=60, top_k=10):
    """Reciprocal Rank Fusion: combine several ranked chunk lists into one
    by summing 1/(k + rank) across every ranker a chunk appears in - a
    chunk one ranker missed but another found still surfaces; a chunk
    both agree on ranks above one only one found. Standard hybrid-search
    technique, k=60 is the commonly-used default. Earlier rankings in the
    list win ties for which chunk dict is kept per id (matters here since
    only the vector ranker's dicts carry a real `distance`)."""
    scores = {}
    chunks_by_id = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id.setdefault(chunk["id"], chunk)
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunks_by_id[cid] for cid in ranked_ids[:top_k]]


def search(query, top_k=5, embedding=None):
    """Exact section-number hits first (if the query names one), then a
    hybrid of vector similarity and BM25 keyword search, reranked by
    Voyage's hosted rerank-2-lite (see rerank.py), fills any remaining
    slots, deduplicated by id. embedding lets a caller reuse an
    already-computed query vector (e.g. Jury RAG embedding once for three
    jurors instead of three times) instead of paying for a fresh Voyage
    call - only valid when the query text is the same one the embedding
    was computed from.

    Vector search's own top hit is always the first non-exact result:
    every pattern's confidence-refusal gate reads chunks[0]["distance"],
    and BM25 scores aren't a comparable distance (no chunk a pure keyword
    search found gets to sit at position 0 and break that comparison with
    None). The hybrid fusion below only improves recall in the remaining
    slots, on top of that unchanged, calibrated gate."""
    exact = exact_section_lookup(query)
    if len(exact) >= top_k:
        return exact[:top_k]

    seen_ids = {c["id"] for c in exact}
    combined = list(exact)

    vec_hits = vector_search(query, top_k=max(top_k, 8), embedding=embedding)
    if not vec_hits:
        return combined

    if vec_hits[0]["id"] not in seen_ids:
        combined.append(vec_hits[0])
        seen_ids.add(vec_hits[0]["id"])

    from bm25_index import bm25_search

    kw_hits = bm25_search(query, top_k=max(top_k, 8))
    fused = _rrf_merge([vec_hits, kw_hits], top_k=top_k * 3)
    candidates = [c for c in fused if c["id"] not in seen_ids]

    if candidates:
        from rerank import rerank

        try:
            candidates = rerank(query, candidates, top_k=len(candidates))
        except Exception:
            # a reranker outage shouldn't take down every pattern's
            # retrieval - fall back to the RRF order rather than crash.
            pass

    for c in candidates:
        if len(combined) >= top_k:
            break
        combined.append(c)
        seen_ids.add(c["id"])
    return combined


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "what is the tax treatment of house rent allowance"
    for c in vector_search(query):
        m = c["metadata"]
        print(f"[{m['act_title']}] Section {m['section']} (dist={c['distance']:.3f})")
        print(f"  {c['text'][:150]}")
