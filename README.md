# Income Tax RAG

A citation-grounded RAG system over Indian income tax law — ask a question and get an answer grounded in the actual statutory text, with a citation back to the exact Act, chapter, and section it came from.

## Why this project

India's **Income-tax Act, 2025** (536 sections, 23 chapters, 16 schedules) came into force on 1 April 2026, completely renumbered from the old 1961 Act (800+ sections, 47 chapters) it replaced. That's after most LLMs' training cutoffs, including this assistant's own (January 2026) — ask a base LLM "what does Section 194 of the new Act say" and it will either confuse it with the old Act's section of the same number, or produce something plausible-sounding and wrong, with no way to tell the difference.

That's the actual value case for RAG here, not a hypothetical one:
- Every answer is grounded in retrieved text from the real Act, cited by chapter/section — verifiable, not "trust the model."
- It sidesteps a knowledge cutoff problem entirely by indexing the current text directly, instead of relying on what a model memorized during training.
- It can say "not found in the indexed source" instead of confidently fabricating a section number — the most dangerous LLM failure mode in a legal/tax context.
- When the Finance Act amends provisions next year, the index gets rebuilt from updated text. No retraining.

## Corpus

Two sources, chosen to support "how did this change from the old law" comparisons:

1. **Income-tax Act, 2025** (Act No. 30 of 2025) — the current law. Source: [ICAI-hosted copy](https://resource.cdn.icai.org/87647dtc-aps2139-inceome-tax-act-2025.pdf), reflecting the Act as originally enacted (August 2025). Note: this predates any Finance Act 2026 amendments — the officially amended version lives at incometaxindia.gov.in but that site blocks scripted access (403 to direct requests, timed out under browser automation too); this is the best available scriptable source at build time.
2. **Income-tax Act, 1961** (Act No. 43 of 1961) — the superseded law, kept for comparison. Source: [India Code bitstream](https://www.indiacode.nic.in/bitstream/123456789/2435/1/a1961-43.pdf). This turned out, once extracted, to be a consolidated text with decades of amendments folded in (e.g. section 92CE, added 2017, and 10BA are both present) rather than the bare 1962-original text the filename suggests — good for the "old law" comparison use case, just not literally frozen at 1961.

Both are government/statutory-body text — public domain under Indian copyright law (Copyright Act 1957, §52(1)(q)).

**Deferred: Income-tax Rules, 1962.** The only scriptable source found (a thc.nic.in "all amendments" bundle) turned out, after actually downloading and inspecting it, to be a chronological stack of amendment notifications and bilingual ITR form schedules — tables and form field layouts, not consolidated rule-by-rule text. Not usable as-is; needs a better source before it's worth adding. See `data/sources.json` for the note.

## How this differs from the other two portfolio RAGs

- [ai-incident-rag](https://github.com/dineshyadav03/ai-incident-rag) chunks hand-curated prose (incident postmortems) by markdown section, cites a source URL, uses a general-purpose sentence embedding model.
- This project chunks **statutory PDF text** by chapter/section numbering (a different, messier extraction problem — PDFs, not clean markdown), cites **Act + chapter + section** instead of a URL, and the core deliverable is the citation grounding itself, motivated by a demonstrated, dated LLM knowledge gap rather than a general RAG demo.

## Status

Phase 1 core pipeline built and verified end-to-end (2026-08-09).

**Extraction**: 536/536 sections recovered for the Income-tax Act, 2025 (100%); 707 sections for the Income-tax Act, 1961. Section boundaries are detected from PDF font metadata (real section numbers are bold; Schedule/table row numbers aren't) rather than pattern-matching flattened text — a naive regex pass was tried first and produced 5,471 false-positive "sections" for the 1961 Act alone, so this isn't a minor implementation detail, it's the reason extraction works at all. Getting to 100% took several rounds of finding and fixing silent data loss: two different fixed max-length filters (8000, then 40000 chars) each dropped real, heavily-amended sections before being caught by manually checking specific missing sections against the source PDF (Section 139 of the 1961 Act, "Return of income," is 43,850 characters after 60+ years of Finance Act amendments — no fixed ceiling is a sound heuristic for legal text length).

**Retrieval**: vector search (ChromaDB + all-MiniLM-L6-v2) plus a direct exact-section-number lookup, added after verifying the core failure mode by hand — a generic query like "what does section 194 say" put the true Section 194 at cosine distance 1.18, while an unrelated section that merely *mentions* "194" in a cross-reference list scored 0.61 and won. Any query naming a section number now gets that section by metadata filter (guaranteed match) before vector search fills remaining slots. Topical queries that don't name a section (e.g. "who has to file a return") still rely on pure semantic similarity and can miss the best section — the same gap [ai-incident-rag](https://github.com/dineshyadav03/ai-incident-rag) solved with hybrid BM25+vector search, not yet ported here.

**Generation**: Groq free tier (llama-3.1-8b-instant), grounded prompt with inline citations, refuses when the top match is a poor semantic fit rather than answering from an irrelevant section.

**Known limitations, stated plainly rather than glossed over**:
- Occasional margin-note text bleeds into body text mid-section (the two-column Gazette layout's caption column isn't always fully excluded) — e.g. one verified case reads "...no loss which has return for losses. not been determined..." where "return for losses" is a stray caption fragment.
- Table-heavy content (rate schedules, cross-reference tables) doesn't extract cleanly to prose. Verified case: the 2025 Act's Section 194 references a rate table whose rows extracted as repetitive garbled text, which caused the LLM to loop on a nonsense repeated-"194" response when that section was retrieved. The underlying data issue, not a retrieval or generation-code bug.
- Topical (non-section-number) queries can miss the best-matching section — see Retrieval above.
- Income-tax Rules, 1962 is deferred entirely (see Corpus section) — no procedural/forms coverage yet.

## Planned phases

1. ~~**Core pipeline**~~ — done (see Status)
2. **Old-vs-new comparison** — a query mode that explicitly retrieves from both corpora and diffs them, rather than relying on the LLM to notice from whichever sections happen to be retrieved
3. **Evaluation** — hand-verified Q&A set against real section text; retrieval-hit-rate at minimum
4. **Production-hardening** — hybrid BM25+vector retrieval (fixes the topical-query gap above), table-aware extraction, revisit what else is worth replicating from ai-incident-rag (observability, CI gates, guardrails) vs. genuinely different for a statutory corpus

Zero monetary budget — free/local resources only (local embeddings, Groq's free tier for generation).
