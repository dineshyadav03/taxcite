"""Build data/processed/correspondence_map.json for Correspondence RAG.

Answers "what's the 2025-Act equivalent of this 1961-Act provision" - a
real question the project's own premise (the Acts were renumbered, not
just re-dated) makes hard, and nothing else in the system answers.

Three steps, in increasing cost:
  1. Pick which ~50 sections to map: in-degree in the existing
     cross-reference graph (data/processed/xref_graph.json), a
     data-driven proxy for "foundational provision" - sections other
     parts of the law depend on most - rather than guessing "famous"
     section numbers, which the renumbering makes unreliable to assume.
  2. Generate 2025-Act candidates with ZERO new embedding calls: fetch
     ALL of a section's already-computed chunk vectors straight out of
     ChromaDB, mean-pool them into one representative vector, and query
     the collection filtered to itact2025 with it. Reuses vectors already
     paid for during the original index build, which is what keeps this
     fast despite Voyage's 3 RPM free-tier cap. Mean-pooling matters for
     the ~25 of 50 top-in-degree sections that run to 4-73 chunks
     ("omnibus" sections like Section 2's definitions or Section 10's
     exemptions list) - verified live: pooling only the FIRST chunk's
     vector for Section 2 (definitions) surfaced "strong match: 2025
     Section 405 (advance tax computation)", because chunk 0 happens to
     open with the "advance tax" definition - a real, section-scope
     mismatch the mean-pooled vector (and the wider text sample below)
     both correct.
  3. One Groq call per section (~50 total, not rate-limited the way
     Voyage is) asking whether each of its top-3 2025 candidates
     functionally corresponds, with a confidence label and rationale.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import graph
from generate import llm
from retrieve import fetch_section, get_collection

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "processed" / "correspondence_map.json"

# llama-3.3-70b-versatile reasons better about the "one item among many in
# an omnibus section" nuance (see generate.llm's docstring) than the
# default 8B model - but verified live: Groq's free tier caps it at a
# hard 100K tokens/DAY, which a ~50-call run alone can exhaust (it did,
# twice). Defaulting to the everyday model (GROQ_MODEL, no override)
# keeps this build inside the zero-budget free tier; the one residual
# miscalibration this leaves (a genuinely-related-but-partial-scope
# candidate occasionally labeled "strong" instead of "partial") is a
# confidence-label imprecision, not a fabricated citation, and is exactly
# what the pattern's fixed disclaimer already covers.
VERIFY_MODEL = None

TOP_N_SECTIONS = 50
CANDIDATES_PER_SECTION = 3
_QUERY_N_RESULTS = 15  # over-fetch, then dedupe to distinct sections
_MAX_CHUNKS_FOR_TEXT = 6  # cap how many of a section's chunks feed the verifier prompt
_CHARS_PER_CHUNK = 600
_CALL_SPACING_SECONDS = 1.5  # light pacing against Groq free-tier bursts

_VERIFY_SYSTEM = """You are verifying whether provisions of India's Income-tax Act, 1961 and its 2025 replacement functionally correspond. You will be given a text sample from one 1961-Act section (abridged with "..." if the section is long) and up to three candidate 2025-Act sections. For EACH candidate, decide if it is the functional equivalent or successor provision to the 1961 section (same substantive rule or overall subject, even if renumbered or reworded).

Caution: some 1961-Act sections are broad omnibus sections - either a glossary defining dozens of unrelated terms in one section (e.g. "advance tax", "agricultural income", "assessee"), or a long list of many distinct exemptions/items grouped under one section (e.g. "incomes not included in total income" covering dozens of unrelated exemption clauses). If the 1961 text sample is clearly this kind of broad, multi-item section, do NOT mark a candidate "strong" just because it narrowly matches or covers ONE of the many terms/items - a broad omnibus section only "strongly" corresponds to another section of comparable breadth covering (most of) the same set of items, not to a narrow provision that only covers one item from the larger list.

Reply with JSON: {"candidates": [{"act_2025_section": "<the section number given>", "confidence": "strong"|"partial"|"none", "rationale": "one sentence"}]}. Include one entry per candidate given, in the same order. "strong" = clearly the same provision (or, for glossary sections, another glossary section of comparable breadth). "partial" = related/overlapping but not a clean match. "none" = unrelated despite being a nearest vector match."""


def top_sections_by_in_degree(act_id, n):
    g = graph.get_graph()
    in_degree = Counter()
    known_sections = set()
    for key, refs in g.items():
        src_act, src_section = key.split("::", 1)
        if src_act == act_id:
            known_sections.add(src_section)
        for ref in refs:
            if src_act == act_id:
                in_degree[ref] += 1
    ranked = sorted(in_degree.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def section_mean_embedding(collection, act_id, section):
    """Mean-pool every chunk vector of a section into one representative
    vector - a section's first chunk alone can badly under-represent its
    scope (an omnibus section's opening lines are one topic among many)."""
    fetched = collection.get(
        where={"$and": [{"section": section}, {"act_id": act_id}]},
        include=["embeddings"],
    )
    if not fetched["ids"]:
        return None
    return np.mean(np.array(fetched["embeddings"]), axis=0).tolist()


def candidates_for_section(collection, act_id, section, k):
    embedding = section_mean_embedding(collection, act_id, section)
    if embedding is None:
        return []
    results = collection.query(
        query_embeddings=[embedding],
        n_results=_QUERY_N_RESULTS,
        where={"act_id": "itact2025"},
    )
    best_distance = {}
    for i in range(len(results["ids"][0])):
        cand_section = results["metadatas"][0][i]["section"]
        distance = results["distances"][0][i]
        if cand_section not in best_distance or distance < best_distance[cand_section]:
            best_distance[cand_section] = distance
    ranked = sorted(best_distance.items(), key=lambda kv: kv[1])
    return [section for section, _ in ranked[:k]]


def section_text(act_id, section):
    """A bounded but representative sample of a section's text: a capped
    slice of EVERY chunk (up to _MAX_CHUNKS_FOR_TEXT of them), not just a
    literal prefix of the concatenated whole - a flat prefix truncation
    starves omnibus sections down to whatever topic happens to open the
    section, which the verifier LLM then judges against."""
    chunks = fetch_section(act_id, section)
    sample = chunks[:_MAX_CHUNKS_FOR_TEXT]
    text = "\n".join(c["text"][:_CHARS_PER_CHUNK] for c in sample)
    if len(chunks) > _MAX_CHUNKS_FOR_TEXT:
        text += f"\n... [{len(chunks) - _MAX_CHUNKS_FOR_TEXT} more chunks omitted - this section runs to {len(chunks)} chunks total]"
    return text


def verify(section_1961, candidates_2025):
    text_1961 = section_text("itact1961", section_1961)
    listing = "\n\n".join(
        f"Candidate {i} - 2025 Act Section {c}:\n{section_text('itact2025', c)}"
        for i, c in enumerate(candidates_2025)
    )
    prompt = f"1961 Act Section {section_1961}:\n{text_1961}\n\n---\n\n{listing}"
    try:
        verdict = llm(prompt, system_prompt=_VERIFY_SYSTEM, json_mode=True, max_tokens=600, model=VERIFY_MODEL)
    except ValueError as e:
        print(f"  section {section_1961}: LLM verification failed ({e}), skipping")
        return []
    raw_candidates = verdict.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return []
    out = []
    for i, c in enumerate(candidates_2025):
        match = next(
            (rc for rc in raw_candidates if isinstance(rc, dict) and str(rc.get("act_2025_section")) == str(c)),
            raw_candidates[i] if i < len(raw_candidates) and isinstance(raw_candidates[i], dict) else {},
        )
        out.append(
            {
                "act_2025_section": c,
                "confidence": str(match.get("confidence", "none")).lower().strip(),
                "rationale": match.get("rationale", ""),
            }
        )
    return out


def main(resume=True):
    """Resumable and incrementally written, same discipline as embed.py's
    build_index: the 70B verify model's free-tier daily token quota is
    small enough (100K TPD) that a 50-call run can hit it mid-run
    (verified live: it did, at call 44) - losing 43 already-paid-for
    calls to a crash, or re-spending tokens re-verifying them on rerun,
    would both be avoidable."""
    collection = get_collection()
    correspondence_map = {}
    if resume and OUT_PATH.exists():
        correspondence_map = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    ranked = top_sections_by_in_degree("itact1961", TOP_N_SECTIONS)
    print(f"Top {len(ranked)} sections by in-degree (foundational provisions):")
    for section, degree in ranked[:10]:
        print(f"  Section {section}: cited by {degree} other sections")
    if correspondence_map:
        print(f"Resuming: {len(correspondence_map)} sections already mapped, skipping those.")

    for i, (section, degree) in enumerate(ranked):
        if section in correspondence_map:
            continue
        if correspondence_map:
            time.sleep(_CALL_SPACING_SECONDS)
        candidates = candidates_for_section(collection, "itact1961", section, CANDIDATES_PER_SECTION)
        if not candidates:
            print(f"  [{i + 1}/{len(ranked)}] Section {section}: no chunk indexed, skipping")
            continue
        verified = verify(section, candidates)
        correspondence_map[section] = {
            "act_1961_section": section,
            "in_degree": degree,
            "candidates": verified,
        }
        OUT_PATH.write_text(json.dumps(correspondence_map, indent=1), encoding="utf-8")
        strong = [c["act_2025_section"] for c in verified if c["confidence"] == "strong"]
        print(f"  [{i + 1}/{len(ranked)}] Section {section} -> strong matches: {strong or 'none'}")

    print(f"\nWrote {len(correspondence_map)} entries -> {OUT_PATH}")


if __name__ == "__main__":
    main(resume="--fresh" not in sys.argv)
