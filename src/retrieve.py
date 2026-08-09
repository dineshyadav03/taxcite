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
from pathlib import Path

import chromadb

from embed import get_embedding_model

ROOT = Path(__file__).resolve().parent.parent

_collection = None

_SECTION_QUERY_RE = re.compile(r"\bsection\s+(\d{1,4}[A-Z]{0,4}(?:-[A-Z0-9]{1,4})?)\b", re.IGNORECASE)
_ACT_2025_RE = re.compile(r"\b(2025|new)\s+act\b", re.IGNORECASE)
_ACT_1961_RE = re.compile(r"\b(1961|old)\s+act\b", re.IGNORECASE)


def get_collection(persist_dir=None):
    global _collection
    if _collection is None:
        persist_dir = str(persist_dir or ROOT / "chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        _collection = client.get_collection("income_tax_sections")
    return _collection


def vector_search(query, top_k=5):
    collection = get_collection()
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

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
    chunks.sort(key=lambda c: (c["metadata"]["act_id"], c["metadata"]["chunk_index"]))
    return chunks


def search(query, top_k=5):
    """Exact section-number hits first (if the query names one), then
    vector search fills any remaining slots, deduplicated by id."""
    exact = exact_section_lookup(query)
    if len(exact) >= top_k:
        return exact[:top_k]

    seen_ids = {c["id"] for c in exact}
    combined = list(exact)
    for c in vector_search(query, top_k=top_k):
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
