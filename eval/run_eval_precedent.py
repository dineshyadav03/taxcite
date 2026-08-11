"""Evaluate Precedent RAG against a small golden set of questions that
should retrieve a section with real linked case law.

Ground truth is derived from data/processed/cases.jsonl's own already-
verified case-to-section links (see README's "Three inventions" section
for how those links were manually confirmed against real section text) -
each question targets one of the 16 linked sections, phrased close to
the case's actual holding so retrieval should land on the right section.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from patterns import run as run_pattern  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_golden():
    return json.loads((ROOT / "eval" / "golden_qa_precedent.json").read_text(encoding="utf-8"))


def run():
    golden = load_golden()
    results = []

    for item in golden:
        result = run_pattern("precedent", item["question"])
        cited = item["expected_case"].lower() in result["answer"].lower()
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_case": item["expected_case"],
            "refused": result["refused"],
            "cited_expected_case": cited,
            "pass": cited and not result["refused"],
        }
        results.append(row)
        print(f"[{'PASS' if row['pass'] else 'FAIL'}] {item['id']}: expected '{item['expected_case']}'")

    passed = sum(1 for r in results if r["pass"])
    accuracy = passed / len(results)
    print(f"\n{passed}/{len(results)} passed ({accuracy * 100:.0f}%)")

    out_path = ROOT / "eval" / "results_precedent.json"
    out_path.write_text(json.dumps({"results": results, "accuracy": accuracy}, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
