"""Evaluate Correspondence RAG against a small golden set of known-good
1961-to-2025 section mappings.

Ground truth is sourced directly from data/processed/correspondence_map.json's
own "strong"-confidence entries - the same ones already manually verified
against real section text during the original build (see README's
"Three inventions" section) - not fabricated for this eval. The one
known-borderline entry (Section 10 -> 338, disclosed in the README as a
"strong" label that should really be "partial") is deliberately excluded,
so this eval measures whether the live pattern reproduces its own map
correctly, not whether the map itself is perfect - that's a different,
already-documented question.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from patterns import run as run_pattern  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_golden():
    return json.loads((ROOT / "eval" / "golden_qa_correspondence.json").read_text(encoding="utf-8"))


def run():
    golden = load_golden()
    results = []

    for item in golden:
        question = f"What's the 2025-Act equivalent of Section {item['section_1961']} of the 1961 Act?"
        result = run_pattern("correspondence", question)
        pattern = re.compile(rf"\bsection\s+{re.escape(item['expected_2025_section'])}\b", re.IGNORECASE)
        cited = bool(pattern.search(result["answer"]))
        row = {
            "id": item["id"],
            "section_1961": item["section_1961"],
            "expected_2025_section": item["expected_2025_section"],
            "refused": result["refused"],
            "cited_expected": cited,
            "pass": cited and not result["refused"],
        }
        results.append(row)
        print(f"[{'PASS' if row['pass'] else 'FAIL'}] {item['id']}: 1961 s{item['section_1961']} -> expected 2025 s{item['expected_2025_section']}")

    passed = sum(1 for r in results if r["pass"])
    accuracy = passed / len(results)
    print(f"\n{passed}/{len(results)} passed ({accuracy * 100:.0f}%)")

    out_path = ROOT / "eval" / "results_correspondence.json"
    out_path.write_text(json.dumps({"results": results, "accuracy": accuracy}, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
