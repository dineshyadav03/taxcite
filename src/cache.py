"""Response cache for /api/ask - exact-match first, semantic second.

Two tiers, deliberately different cost profiles:

1. Exact match (SQLite, `data/cache.db`): a plain string lookup on
   (pattern, normalized question). Zero API cost to check, zero latency.
   This is what protects repeated dev/eval traffic - re-running the same
   golden-set question against the same pattern, which is most of what
   actually repeats during this project's own testing.

2. Semantic match (a small in-process ChromaDB collection of past
   question embeddings): catches paraphrases ("who has to file a
   return" vs "who is required to file an income tax return") that
   exact-match can't. Real tradeoff, stated plainly rather than oversold:
   checking it costs ONE Voyage embedding call even on a miss, since the
   incoming question has to be embedded to compare against past ones.
   That's a net win for real user-facing traffic (a hit saves the
   pattern's own embedding + generation cost entirely), but it is not
   free insurance - on a cold cache (e.g. a fresh eval run of 20 unique
   questions), it adds one extra embedding call per question with no
   corresponding hit. Exact-match has no such cost either way, which is
   why it's checked first and alone protects the case that matters most
   for this project's own quota-fragile testing.

Cache entries are never invalidated by TTL - the underlying corpus only
changes when the index is rebuilt, at which point clearing data/cache.db
is the right move (not built here, since this project's own build
scripts already aren't run casually - see scripts/build_chunks.py etc).
"""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import chromadb

from retrieve import embed_query

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB_PATH = ROOT / "data" / "cache.db"
CACHE_CHROMA_DIR = ROOT / "data" / "cache_chroma"

# Cosine distance below which a past question counts as "the same
# question" for cache purposes. NOT the same measurement space as
# DISTANCE_REFUSAL_THRESHOLD (1.15) - that's query-to-document (Voyage's
# asymmetric encoding), this is query-to-query (symmetric, same
# input_type="query" on both sides), so the two thresholds aren't
# comparable and reusing 1.15 here would be a guess dressed up as reuse.
# Calibrated live instead: two genuine paraphrases of the same question
# ("What is the Site Restoration Fund provision under the 1961 Act?" vs.
# "What does the 1961 Act say about the Site Restoration Fund?" / "Tell
# me about Site Restoration Fund rules in Indian tax law") measured
# 0.466 and 0.281; two genuinely different questions measured 1.451 and
# 1.964. Clear gap between ~0.47 and ~1.45 - 0.6 sits in it with real
# margin on both sides, biased toward the tighter end on purpose: a
# false-positive cache hit (serving a wrong cached answer) is a worse
# failure than a false negative (falling through to a correct, fresh
# computation), so this errs toward missing a real match over risking
# a wrong one.
_SEMANTIC_CACHE_THRESHOLD = 0.6

_sqlite_conn = None
_chroma_collection = None


def _normalize(question):
    return re.sub(r"\s+", " ", question.strip().lower())


def _get_sqlite():
    global _sqlite_conn
    if _sqlite_conn is None:
        CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _sqlite_conn = sqlite3.connect(str(CACHE_DB_PATH), check_same_thread=False)
        _sqlite_conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  pattern TEXT NOT NULL,"
            "  question_key TEXT NOT NULL,"
            "  response_json TEXT NOT NULL,"
            "  created_at TEXT NOT NULL,"
            "  PRIMARY KEY (pattern, question_key)"
            ")"
        )
        _sqlite_conn.commit()
    return _sqlite_conn


def _get_chroma():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CACHE_CHROMA_DIR))
        existing = [c.name for c in client.list_collections()]
        _chroma_collection = (
            client.get_collection("query_cache") if "query_cache" in existing else client.create_collection("query_cache")
        )
    return _chroma_collection


def get(pattern, question):
    """Returns (response_or_None, embedding_or_None). The embedding is
    the query vector computed during the semantic check, when one was
    needed - callers pass it to put() on a miss so the write doesn't pay
    for a second, redundant Voyage call for the exact same question text.
    Only a genuine cache miss ever returns a non-None embedding; an exact
    hit or an empty cache short-circuits before any embedding call."""
    conn = _get_sqlite()
    key = _normalize(question)
    row = conn.execute(
        "SELECT response_json FROM cache WHERE pattern = ? AND question_key = ?", (pattern, key)
    ).fetchone()
    if row:
        response = json.loads(row[0])
        response["cached"] = "exact"
        return response, None

    collection = _get_chroma()
    if collection.count() == 0:
        return None, None

    embedding = embed_query(question)
    results = collection.query(query_embeddings=[embedding], n_results=1, where={"pattern": pattern})
    if not results["ids"][0]:
        return None, embedding
    distance = results["distances"][0][0]
    if distance > _SEMANTIC_CACHE_THRESHOLD:
        return None, embedding

    cache_key = results["metadatas"][0][0]["question_key"]
    row = conn.execute(
        "SELECT response_json FROM cache WHERE pattern = ? AND question_key = ?", (pattern, cache_key)
    ).fetchone()
    if not row:
        return None, embedding  # stale chroma entry with no matching sqlite row - treat as a miss
    response = json.loads(row[0])
    response["cached"] = "semantic"
    response["cache_distance"] = round(distance, 3)
    return response, None


def put(pattern, question, response, created_at, embedding=None):
    """Persists a fresh response under both tiers. response should be the
    raw dict about to be returned to the client, before any "cached" key
    is added - store what a fresh answer actually looked like. Pass the
    embedding get() returned on the miss that led here, if any, so this
    doesn't re-embed the same question text a second time."""
    conn = _get_sqlite()
    key = _normalize(question)
    conn.execute(
        "INSERT OR REPLACE INTO cache (pattern, question_key, response_json, created_at) VALUES (?, ?, ?, ?)",
        (pattern, key, json.dumps(response), created_at),
    )
    conn.commit()

    if embedding is None:
        embedding = embed_query(question)
    collection = _get_chroma()
    entry_id = hashlib.sha256(f"{pattern}::{key}".encode()).hexdigest()
    collection.upsert(
        ids=[entry_id],
        embeddings=[embedding],
        metadatas=[{"pattern": pattern, "question_key": key}],
    )
