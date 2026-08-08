# Codebase RAG

A citation-grounded RAG system over a real, in-production open-source codebase (FastAPI) — ask "how does X work?" and get an answer grounded in the actual source, with a citation back to the exact file and line it came from.

## Status

Just started (2026-08-08). Scope and phase plan below; nothing built yet.

## Why this project, and how it differs from [ai-incident-rag](https://github.com/dineshyadav03/ai-incident-rag)

`ai-incident-rag` is a RAG system over hand-curated prose (incident postmortems) — chunked by markdown section, embedded with a general-purpose sentence embedding model. This project is deliberately a different technical problem, not the same architecture with a new topic:

- **Chunking by code structure** (function/class boundaries, or AST-aware) instead of prose section headers — a real, untested-until-now comparison against naive fixed-size chunking.
- **Code-oriented embeddings**, benchmarked against a general-purpose model rather than assumed.
- **Citations are file path + line number**, not a source URL — a different grounding contract entirely.
- Retrieval likely needs an exact-symbol-name boost (a function called `add_middleware` should be found by that literal name, which general embedding similarity doesn't always guarantee) alongside semantic search.

## Target corpus

[FastAPI](https://github.com/fastapi/fastapi) — pinned to a specific tag/commit for reproducibility (TBD). Chosen for recognizability, moderate size, and clean structure.

## Planned phases

1. **Core pipeline** — vendor a pinned FastAPI source snapshot, chunk by code structure, embed, index, retrieve, generate with file+line citations
2. **Chunking + embedding comparison** — the two things `ai-incident-rag` never actually tested: fixed-size vs. structure-aware chunking, general-purpose vs. code-oriented embeddings, measured not assumed
3. **Evaluation** — a hand-verified golden Q&A set about FastAPI internals, retrieval-hit-rate at minimum, RAGAS-style scoring if it adds real signal for code (TBD — RAGAS's metrics are tuned for prose, may not transfer cleanly)
4. **Production-hardening** — revisit what actually made sense to replicate from `ai-incident-rag` (observability, CI gates, guardrails) vs. what's genuinely different for a code corpus

Zero monetary budget — same constraint as `ai-incident-rag`: free/local resources only (local Ollama or Groq's free tier for generation, free-tier compute for anything heavier).
