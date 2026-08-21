"""Shared chunk-relevance grading - originally Corrective RAG's own
private step, extracted so every pattern can use the same check.

The failure this catches: vector search always returns *something* -
its top-k is "the least-distant chunks", not "relevant chunks". One
grading call classifies every retrieved chunk as relevant/irrelevant to
the actual question; only graded-relevant chunks reach the generator,
so an off-target chunk can't quietly steer the answer.

This module is deliberately just the grading call, not a retry loop.
Corrective RAG keeps its own rewrite-and-retry behavior on top of this
(see patterns/corrective.py) - that's its real differentiator from
every other pattern that now also grades. A pattern using grade_chunks()
directly (Simple, Graph, Memory, Multimodal) grades once and, on a
grader failure or an empty result, is expected to make its own honest
call about what to do next (degrade to ungraded chunks on a parse
failure, refuse on a genuine zero-relevant result) rather than retry -
see each pattern's own answer() for how it handles that.
"""
import re

from generate import llm

_GRADE_SYSTEM = """You grade retrieved statute excerpts for relevance to a question. For each excerpt decide: is it on-topic - does it discuss the section, provision, or subject the question is actually about? Mark an excerpt relevant if it would be USEFUL as part of an answer, even if it only covers one part of a multi-part or comparison question (e.g. an excerpt about one of two Acts being compared is still relevant, even though it alone can't answer the whole comparison). Only mark an excerpt irrelevant if it is genuinely off-topic - a different section or subject with no bearing on the question. An excerpt marked [REPEALED/OMITTED SECTION] is no longer in force - its text is just a repeal stub, and any other-looking content after it is bleed-through from the next section's heading, not that section's actual content. Mark a repealed excerpt irrelevant unless the question specifically asks about a repealed, omitted, or historical provision. Reply with JSON: {"grades": [{"index": 0, "relevant": true/false, "reason": "..."}], "better_query": "an improved search query if most excerpts are irrelevant, else null"}."""

# A repealed section's stored text is just a repeal stub - "86A. [...]
# Omitted by the Finance Act, 1988 ..." - followed by whatever heading
# text happens to sit next in the source PDF (e.g. the next chapter's
# title), a known extraction artifact (see project memory: "margin-note
# bleed-through"). That bleed-through can contain real keywords ("Rebate
# of income-tax") that make an omitted section look like a strong match
# for an unrelated live question - verified live: Section 86A (omitted
# 1988) out-competed the actual, currently-in-force Section 87 for "what
# provides a rebate in computing income-tax" because 86A's stub happens
# to butt up against a "REBATES AND RELIEFS" chapter heading. Flagging it
# explicitly here is a runtime fix (no re-embedding needed) - a real fix
# at the extraction source is a separate, larger effort.
_OMITTED_RE = re.compile(r"\bomitted by the\b", re.IGNORECASE)


def grade_chunks(question, chunks):
    """Returns (relevant_chunks, grades_by_index, better_query_or_none).
    Raises ValueError if the grader's output can't be parsed (matches
    every other json_mode caller's contract in this codebase - callers
    are expected to catch this and degrade, not let it propagate)."""
    if not chunks:
        return [], {}, None
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('act_title')}, Section {c['metadata'].get('section')})"
        f"{' [REPEALED/OMITTED SECTION]' if _OMITTED_RE.search(c['text'][:150]) else ''}\n{c['text'][:500]}"
        for i, c in enumerate(chunks)
    )
    verdict = llm(
        f"Question: {question}\n\nExcerpts:\n{listing}",
        system_prompt=_GRADE_SYSTEM,
        json_mode=True,
        # Raised from 400 after the Groq model swap (llama-3.1-8b-instant
        # removed platform-side, replaced with openai/gpt-oss-20b) - the
        # new model spends part of its token budget on internal reasoning
        # before writing visible output, verified live: a trivial 5-token
        # budget returned nothing at all, not an error, just empty. 400
        # was sized for the old model's direct-answer behavior and left
        # no headroom for that, especially grading up to 7 chunks at once.
        max_tokens=1200,
    )
    grades = {g["index"]: g for g in verdict.get("grades", []) if isinstance(g, dict) and "index" in g}
    relevant = [c for i, c in enumerate(chunks) if grades.get(i, {}).get("relevant")]
    return relevant, grades, verdict.get("better_query")


def grading_trace_entry(question, chunks, grades):
    """The same {section, act, relevant} trace shape every pattern that
    grades should show, factored out so it's identical everywhere rather
    than each pattern reinventing its own slightly different version."""
    return {
        "query": question,
        "graded": [
            {
                "section": c["metadata"].get("section"),
                "act": c["metadata"].get("act_id"),
                "relevant": grades.get(i, {}).get("relevant", False),
            }
            for i, c in enumerate(chunks)
        ],
    }
