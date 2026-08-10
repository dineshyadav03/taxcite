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

    act_filter = None
    if _ACT_2025_RE.search(query):
        act_filter = "itact2025"
    elif _ACT_1961_RE.search(query):
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


def search(query, top_k=5, embedding=None):
    """Exact section-number hits first (if the query names one), then
    vector search fills any remaining slots, deduplicated by id.
    embedding lets a caller reuse an already-computed query vector
    (e.g. Jury RAG embedding once for three jurors instead of three
    times) instead of paying for a fresh Voyage call - only valid when
    the query text is the same one the embedding was computed from."""
    exact = exact_section_lookup(query)
    if len(exact) >= top_k:
        return exact[:top_k]

    seen_ids = {c["id"] for c in exact}
    combined = list(exact)
    for c in vector_search(query, top_k=top_k, embedding=embedding):
        if len(combined) >= top_k:
            break
        if c["id"] not in seen_ids:
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
