"""Sub-chunk long sections and embed everything into a ChromaDB collection.

Embeddings come from Voyage AI's hosted embedding API (voyage-4, free tier
- see README) rather than a local model: this project's zero-budget
constraint prefers free *hosted* compute over local when local isn't
actually needed, and embedding an entire corpus is exactly the kind of
batch job that's better off a personal machine.

Voyage AI was picked over Google's Gemini Embedding API after actually
hitting Gemini's free-tier ceiling live: 1,000 requests/day
(EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier), discovered
only from the full error detail after several rounds of assuming it was a
transient rate limit and adding backoff/pacing - it wasn't, it was a hard
daily cap smaller than this corpus (2,402 chunks). Voyage's free tier is
200M tokens (about 100-200x this corpus's total token count), which
avoids that failure mode outright rather than working around it.

data/processed/*.jsonl is one row per real Act section (the citation
unit). Some sections run to 10k+ characters, too large for a single
embedding to stay semantically focused - so at index time (not extraction
time) we split anything over the token budget into overlapping windows.
Every resulting chunk keeps its parent section/act/title in metadata, so
retrieval always cites at the section level regardless of which window
matched.
"""
import json
import os
import time
from pathlib import Path

import chromadb
import tiktoken
import voyageai
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_NAME = "voyage-4"
CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100
# Voyage's free tier without a payment method on file is throttled hard:
# 3 RPM and 10K TPM (confirmed live - the standard-tier batch size/pacing
# 429'd immediately). A payment method would lift this, but that's a real
# ask this project deliberately avoids (zero monetary budget, no payment
# info entered anywhere) even though usage itself stays free either way.
# Batch size is capped so even a full batch can't exceed 10K TPM
# (10 * 700 max tokens/chunk = 7000, safely under), and spacing matches
# the 3 RPM ceiling (20s cadence + a buffer).
EMBED_BATCH_SIZE = 10
EMBED_MAX_RETRIES = 5
EMBED_CALL_SPACING_SECONDS = 21

_encoding = tiktoken.get_encoding("cl100k_base")
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY not set. Get a free key at https://www.voyageai.com "
                "and add it to .env (no credit card required, 200M free tokens)."
            )
        # No timeout is set by default, so a stalled connection hangs
        # indefinitely instead of failing into our own retry loop -
        # verified live: a build hung silently for 20+ minutes with the
        # process still alive, no crash, no progress (network itself was
        # fine when checked separately - the request just never returned
        # or errored on its own).
        _client = voyageai.Client(api_key=api_key, timeout=30)
    return _client


def embed_texts(texts, input_type):
    """Embed a list of texts via Voyage AI, batched with retry-with-backoff
    and steady call spacing (a request-rate limit needs pacing, not just
    backoff after failure - learned the hard way with Gemini's endpoint)."""
    client = get_client()
    all_embeddings = []
    num_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_num, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE)):
        if batch_num > 0:
            time.sleep(EMBED_CALL_SPACING_SECONDS)
        batch = texts[start : start + EMBED_BATCH_SIZE]
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = client.embed(batch, model=EMBEDDING_MODEL_NAME, input_type=input_type)
                all_embeddings.extend(response.embeddings)
                break
            except Exception as e:
                if attempt == EMBED_MAX_RETRIES - 1:
                    raise
                wait = EMBED_CALL_SPACING_SECONDS * (2 ** (attempt + 1))
                print(f"  embed batch failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        if len(texts) > EMBED_BATCH_SIZE:
            print(f"  embedded batch {batch_num + 1}/{num_batches}")
    return all_embeddings


def split_long_text(text, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    tokens = _encoding.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    pieces = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        pieces.append(_encoding.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return pieces


def load_sections():
    for path in sorted((ROOT / "data" / "processed").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def build_index(persist_dir=None, resume=True):
    """Resumable by default: if a prior run died partway through, rerunning
    skips chunk ids already in the collection instead of re-embedding them.
    Pass resume=False to force a clean rebuild (required when switching
    embedding providers - vectors from different models aren't comparable,
    so a collection built with a prior provider must be wiped, not resumed)."""
    persist_dir = str(persist_dir or ROOT / "chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    existing_names = [c.name for c in client.list_collections()]
    if not resume and "income_tax_sections" in existing_names:
        client.delete_collection("income_tax_sections")
        existing_names.remove("income_tax_sections")
    collection = (
        client.get_collection("income_tax_sections")
        if "income_tax_sections" in existing_names
        else client.create_collection("income_tax_sections")
    )
    already_done = set(collection.get(include=[])["ids"]) if resume else set()

    documents, metadatas, ids = [], [], []
    for section in load_sections():
        pieces = split_long_text(section["text"])
        for i, piece in enumerate(pieces):
            chunk_id = f"{section['act_id']}::{section['section']}::{i}"
            if chunk_id in already_done:
                continue
            documents.append(piece)
            metadatas.append(
                {
                    "act_id": section["act_id"],
                    "act_title": section["act_title"],
                    "chapter": section.get("chapter") or "",
                    "section": section["section"],
                    "title": section.get("title") or "",
                    "chunk_index": i,
                    "chunk_count": len(pieces),
                }
            )
            ids.append(chunk_id)

    if not documents:
        print(f"Nothing to do - {collection.count()} chunks already indexed in {persist_dir}")
        return collection

    print(
        f"Embedding {len(documents)} chunks via {EMBEDDING_MODEL_NAME} (hosted), "
        f"{len(already_done)} already done..."
    )
    for batch_num, start in enumerate(range(0, len(documents), EMBED_BATCH_SIZE)):
        if batch_num > 0:
            time.sleep(EMBED_CALL_SPACING_SECONDS)
        end = start + EMBED_BATCH_SIZE
        batch_embeddings = embed_texts(documents[start:end], input_type="document")
        collection.add(
            documents=documents[start:end],
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  indexed {min(end, len(documents))}/{len(documents)} (total in collection: {collection.count()})")

    print(f"Indexed {collection.count()} chunks into {persist_dir}")
    return collection


if __name__ == "__main__":
    import sys

    build_index(resume="--fresh" not in sys.argv)
