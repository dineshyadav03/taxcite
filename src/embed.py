"""Sub-chunk long sections and embed everything into a ChromaDB collection.

data/processed/*.jsonl is one row per real Act section (the citation unit).
Some sections run to 10k+ characters, too large for a single embedding to
stay semantically focused - so at index time (not extraction time) we split
anything over the token budget into overlapping windows. Every resulting
chunk keeps its parent section/act/title in metadata, so retrieval always
cites at the section level regardless of which window matched.
"""
import json
from pathlib import Path

import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

_encoding = tiktoken.get_encoding("cl100k_base")
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


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


def build_index(persist_dir=None):
    persist_dir = str(persist_dir or ROOT / "chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    client.delete_collection("income_tax_sections") if "income_tax_sections" in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection("income_tax_sections")

    documents, metadatas, ids = [], [], []
    for section in load_sections():
        pieces = split_long_text(section["text"])
        for i, piece in enumerate(pieces):
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
            ids.append(f"{section['act_id']}::{section['section']}::{i}")

    print(f"Embedding {len(documents)} chunks from {len(ids)} pieces...")
    model = get_embedding_model()
    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True).tolist()

    batch_size = 512
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
    print(f"Indexed {collection.count()} chunks into {persist_dir}")
    return collection


if __name__ == "__main__":
    build_index()
