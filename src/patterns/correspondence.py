"""Pattern 12 - Correspondence RAG: 1961-Act section -> its 2025-Act equivalent.

Answers a question nothing else in the system answers: "what's the
2025-Act equivalent of this 1961-Act provision?" The Acts were renumbered
wholesale, not just re-dated, so this is a real, non-trivial lookup - not
something a section-number regex or plain vector search can answer, since
"Section 139" means completely different things in the two Acts.

Backed by data/processed/correspondence_map.json, built offline by
scripts/build_correspondence.py: the ~50 most-cited ("foundational")
1961-Act sections, each mapped to its top-3 nearest-vector 2025-Act
candidates and LLM-verified for functional correspondence with a
confidence label. This pattern does no LLM verification of its own - it
looks the section up, and if it's not in the curated map, says so rather
than guessing.

Every answer carries a disclaimer: this is an LLM-verified functional
mapping, not an official concordance table. It's a real inferential
judgment call (does provision A functionally replace provision B), not a
pure text-retrieval claim the way every other pattern's citations are.
"""
import json
from pathlib import Path

from generate import build_user_prompt, llm
from retrieve import _SECTION_QUERY_RE, fetch_section

ROOT = Path(__file__).resolve().parent.parent.parent
MAP_PATH = ROOT / "data" / "processed" / "correspondence_map.json"

_MAX_CHUNKS_PER_SECTION = 3
_MAX_TOTAL_CHUNKS = 6

DISCLAIMER = (
    "This is an LLM-verified functional mapping, not an official concordance table - "
    "confirm anything load-bearing independently."
)

_NO_SECTION_MESSAGE = (
    "Correspondence RAG maps a specific Income-tax Act, 1961 section to its "
    "Income-tax Act, 2025 equivalent - name a section number to look one up "
    "(e.g. \"what's the 2025-Act equivalent of Section 139?\")."
)

_NOT_MAPPED_MESSAGE = (
    "Section {section} of the Income-tax Act, 1961 is not in the curated correspondence "
    "map (the ~50 most-cited sections of the 1961 Act) - not yet mapped, rather than a "
    "guess presented as an answer."
)

_CORRESPONDENCE_SYSTEM = """You explain how a provision of India's Income-tax Act, 1961 corresponds to its replacement in the Income-tax Act, 2025. You are given the 1961 section's text and one or more candidate 2025 sections that have been judged (separately, already) to correspond, each with a confidence label ("strong", "partial", or "none") and a one-sentence rationale. Using ONLY the provided text:
- Explain the correspondence, citing real section numbers from both Acts.
- If a candidate's confidence is "partial", say plainly that it's a partial/overlapping match, not a clean equivalent.
- Ignore ("none") confidence candidates in your explanation - they were checked and rejected, not omissions.
- Do not use outside knowledge of tax law."""

_map_cache = None


def _load_map():
    global _map_cache
    if _map_cache is None:
        _map_cache = json.loads(MAP_PATH.read_text(encoding="utf-8")) if MAP_PATH.exists() else {}
    return _map_cache


def answer(question, session_id=None):
    m = _SECTION_QUERY_RE.search(question)
    if not m:
        return {"answer": _NO_SECTION_MESSAGE, "chunks": [], "refused": True, "trace": {"section_query": None}}

    section = m.group(1)
    correspondence_map = _load_map()
    entry = correspondence_map.get(section)

    if not entry:
        return {
            "answer": _NOT_MAPPED_MESSAGE.format(section=section),
            "chunks": [],
            "refused": True,
            "trace": {"section_query": section, "in_map": False},
        }

    source_chunks = fetch_section("itact1961", section)[:_MAX_CHUNKS_PER_SECTION]
    candidates = entry["candidates"]
    usable = [c for c in candidates if c["confidence"] in ("strong", "partial")]

    trace = {
        "section_query": section,
        "in_map": True,
        "in_degree": entry.get("in_degree"),
        "candidates": candidates,
    }

    if not usable:
        return {
            "answer": (
                f"Section {section} of the Income-tax Act, 1961 was checked against its nearest "
                f"2025-Act candidates, but none were verified as a functional match (confidence: "
                f"{', '.join(c['confidence'] for c in candidates) or 'none'}). {DISCLAIMER}"
            ),
            "chunks": source_chunks,
            "refused": False,
            "trace": trace,
        }

    candidate_chunks = []
    for c in usable:
        candidate_chunks.extend(fetch_section("itact2025", c["act_2025_section"])[:_MAX_CHUNKS_PER_SECTION])

    candidate_listing = "\n\n".join(
        f"Candidate: Income-tax Act, 2025, Section {c['act_2025_section']} "
        f"(confidence: {c['confidence']}, rationale: {c['rationale']})"
        for c in usable
    )
    # Capped per-section (not just in total) - a source section alone can run
    # several chunks (verified live: Section 32 is 7), which combined with
    # this pattern's extra candidate_listing block on top of the usual
    # build_user_prompt context pushed a real query over Groq's free-tier
    # 6000 TPM ceiling (a reproducible 413, not a timing collision - traced
    # to this exact section before being capped here).
    all_chunks = (source_chunks + candidate_chunks)[:_MAX_TOTAL_CHUNKS]
    prompt = build_user_prompt(question, all_chunks) + f"\n\n---\n\nVerified candidate assessments:\n{candidate_listing}"
    answer_text = llm(prompt, system_prompt=_CORRESPONDENCE_SYSTEM)
    answer_text = f"{answer_text}\n\n{DISCLAIMER}"

    return {"answer": answer_text, "chunks": all_chunks, "refused": False, "trace": trace}
