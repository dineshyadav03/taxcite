"""Evaluate any pattern against the golden Q&A set.

Originally hardcoded to Simple RAG only. Generalized to run any pattern
through the shared `patterns.run()` entry point, since every pattern
returns the same {answer, chunks, refused} contract regardless of what
it did internally to produce it - the golden set and the three metrics
below don't need to know which pattern they're scoring.

Three metrics, chosen deliberately over generic RAGAS-style faithfulness/
relevancy scores: this project's actual value proposition is citation
correctness (does it point you at the right Act+section), which RAGAS
metrics - tuned for prose faithfulness - don't directly measure. Same
judgment call already flagged in the README before any eval code existed.

- retrieval_hit_rate: for questions with a known expected section, is that
  section present anywhere in the chunks the PATTERN ITSELF actually
  returned (not a separate freestanding search() call) - this matters for
  patterns whose own retrieval differs from plain search(), e.g. Graph's
  one-hop expansion or Multi-modal's second collection.
- citation_accuracy: for questions that got a real (non-refused) answer,
  does the generated answer actually cite the expected section?
- refusal_accuracy: for genuinely out-of-corpus questions, does the system
  correctly refuse instead of confabulating an answer?

Branched, Jury, and Correspondence are NOT in STANDARD_PATTERNS below -
their output shapes don't fit this golden set's per-question expected-
section contract (Branched merges multiple sections by design, Jury
votes across three sub-patterns, Correspondence answers a completely
different question). See eval/run_eval_branched.py, run_eval_jury.py,
run_eval_correspondence.py, run_eval_precedent.py for their own scoring.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from patterns import run as run_pattern  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

STANDARD_PATTERNS = [
    "simple",
    "memory",
    "graph",
    "self_rag",
    "precedent",
    "multimodal",
    "corrective",
    "hyde",
    "agentic",
    "adaptive",
]


def load_golden():
    return json.loads((ROOT / "eval" / "golden_qa.json").read_text(encoding="utf-8"))


def check_retrieval_hit(item, chunks):
    hit_sections = {c["metadata"].get("section") for c in chunks}
    return item["section"] in hit_sections


def check_comparison_coverage(result):
    """A real old-vs-new comparison answer should mention both Acts, not
    just whichever one the underlying retrieval happened to favor -
    tightened from the original loose "did it not refuse" check now that
    Branched RAG (built specifically for this question shape) is fast
    enough to iterate on."""
    text = result["answer"]
    return "1961" in text and "2025" in text


def check_citation_accuracy(item, result):
    if result["refused"]:
        return None  # not applicable
    # citation form is "(Act title, Section N)" per the system prompt, but
    # be lenient - just check "Section N" (or "N" bare after "section")
    # appears anywhere in the answer text.
    pattern = re.compile(rf"\bsection\s+{re.escape(item['section'])}\b", re.IGNORECASE)
    return bool(pattern.search(result["answer"]))


def run(pattern_name, ids=None):
    golden = load_golden()
    if ids:
        wanted = set(ids)
        golden = [item for item in golden if item["id"] in wanted]
    results = []

    for item in golden:
        qtype = item["query_type"]
        row = {"id": item["id"], "question": item["question"], "type": qtype}

        # Each golden question is independent, not a real conversation - a
        # shared/default session_id here would let Memory RAG's history-
        # aware query rewrite see prior, unrelated golden questions as
        # "conversation history" and garble retrieval for everything after
        # the first question. Verified live: 2025-01 passes clean on its
        # own but failed when run through the shared-session eval loop.
        result = run_pattern(pattern_name, item["question"], session_id=f"eval-{item['id']}")

        if qtype == "refusal":
            row["refused"] = result["refused"]
            row["pass"] = result["refused"]
        elif qtype == "comparison":
            row["refused"] = result["refused"]
            row["answer_preview"] = result["answer"][:200]
            covers_both = check_comparison_coverage(result)
            row["covers_both_acts"] = covers_both
            row["pass"] = not result["refused"] and covers_both
        else:
            hit = check_retrieval_hit(item, result.get("chunks", []))
            cited = check_citation_accuracy(item, result)
            row["retrieval_hit"] = hit
            row["citation_accurate"] = cited
            row["refused"] = result["refused"]
            row["pass"] = hit and (cited is True)

        results.append(row)
        print(f"[{'PASS' if row['pass'] else 'FAIL'}] {item['id']}: {item['question'][:70]}")

    scoreable = [r for r in results if r["type"] not in ("comparison",)]
    passed = sum(1 for r in scoreable if r["pass"])
    print(f"\n{pattern_name}: {passed}/{len(scoreable)} passed ({passed / len(scoreable) * 100:.0f}%)")

    # None (not 0.0) when a filtered --ids subset happens to contain no
    # questions of that type - a smoke run of just refusal questions
    # shouldn't print a misleading "0.00" for retrieval/citation it never
    # measured.
    topical_exact = [r for r in results if r["type"] in ("topical", "exact_section")]
    hit_rate = sum(1 for r in topical_exact if r.get("retrieval_hit")) / len(topical_exact) if topical_exact else None
    citation_rate = sum(1 for r in topical_exact if r.get("citation_accurate")) / len(topical_exact) if topical_exact else None
    refusal_rows = [r for r in results if r["type"] == "refusal"]
    refusal_rate = sum(1 for r in refusal_rows if r["pass"]) / len(refusal_rows) if refusal_rows else None

    if hit_rate is not None:
        print(f"retrieval_hit_rate: {hit_rate:.2f}")
    if citation_rate is not None:
        print(f"citation_accuracy: {citation_rate:.2f}")
    if refusal_rate is not None:
        print(f"refusal_accuracy: {refusal_rate:.2f}")

    return {
        "pattern": pattern_name,
        "results": results,
        "retrieval_hit_rate": hit_rate,
        "citation_accuracy": citation_rate,
        "refusal_accuracy": refusal_rate,
        "all_passed": passed == len(scoreable),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="simple",
        help="pattern name to evaluate, or 'all' to run every standard-contract pattern",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="comma-separated golden-question ids to run instead of the full set, "
        "e.g. --ids 1961-01,2025-07,compare-01,refuse-01 (used for the CI smoke gate - "
        "see .github/workflows/eval.yml)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any scoreable question fails, instead of always exiting 0 - "
        "the CI gate needs this to actually block a bad push; local exploratory runs "
        "don't, since a known documented limitation failing isn't a build failure",
    )
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else None

    if args.pattern == "all":
        all_results = {}
        all_passed = True
        for name in STANDARD_PATTERNS:
            print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
            all_results[name] = run(name, ids=ids)
            all_passed = all_passed and all_results[name]["all_passed"]
        out_path = ROOT / "eval" / "results_by_pattern.json"
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
    else:
        result = run(args.pattern, ids=ids)
        all_passed = result["all_passed"]
        out_path = ROOT / "eval" / ("results.json" if args.pattern == "simple" else f"results_{args.pattern}.json")
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    if args.fail_on_regression and not all_passed:
        print("\nFAIL-ON-REGRESSION: at least one question failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
