"""Evaluate income-tax-rag against the golden Q&A set.

Two metrics, chosen deliberately over generic RAGAS-style faithfulness/
relevancy scores: this project's actual value proposition is citation
correctness (does it point you at the right Act+section), which RAGAS
metrics - tuned for prose faithfulness - don't directly measure. Same
judgment call already flagged in the README before any eval code existed.

- retrieval_hit_rate: for questions with a known expected section, is that
  section present anywhere in the top-k retrieved chunks?
- citation_accuracy: for questions that got a real (non-refused) answer,
  does the generated answer actually cite the expected section?
- refusal_accuracy: for genuinely out-of-corpus questions, does the system
  correctly refuse instead of confabulating an answer?
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate import answer_question
from retrieve import search

ROOT = Path(__file__).resolve().parent.parent


def load_golden():
    return json.loads((ROOT / "eval" / "golden_qa.json").read_text(encoding="utf-8"))


def check_retrieval_hit(item, top_k=5):
    chunks = search(item["question"], top_k=top_k)
    hit_sections = {c["metadata"]["section"] for c in chunks}
    return item["section"] in hit_sections


def check_citation_accuracy(item, result):
    if result["refused"]:
        return None  # not applicable
    # citation form is "(Act title, Section N)" per the system prompt, but
    # be lenient - just check "Section N" (or "N" bare after "section")
    # appears anywhere in the answer text.
    pattern = re.compile(rf"\bsection\s+{re.escape(item['section'])}\b", re.IGNORECASE)
    return bool(pattern.search(result["answer"]))


def run():
    golden = load_golden()
    results = []

    for item in golden:
        qtype = item["query_type"]
        row = {"id": item["id"], "question": item["question"], "type": qtype}

        if qtype == "refusal":
            result = answer_question(item["question"])
            row["refused"] = result["refused"]
            row["pass"] = result["refused"]
        elif qtype == "comparison":
            result = answer_question(item["question"])
            row["refused"] = result["refused"]
            row["answer_preview"] = result["answer"][:200]
            row["pass"] = not result["refused"]  # loose check - comparison mode isn't built yet
        else:
            hit = check_retrieval_hit(item)
            result = answer_question(item["question"])
            cited = check_citation_accuracy(item, result)
            row["retrieval_hit"] = hit
            row["citation_accurate"] = cited
            row["refused"] = result["refused"]
            row["pass"] = hit and (cited is True)

        results.append(row)
        print(f"[{'PASS' if row['pass'] else 'FAIL'}] {item['id']}: {item['question'][:70]}")

    scoreable = [r for r in results if r["type"] not in ("comparison",)]
    passed = sum(1 for r in scoreable if r["pass"])
    print(f"\n{passed}/{len(scoreable)} passed ({passed / len(scoreable) * 100:.0f}%)")

    topical_exact = [r for r in results if r["type"] in ("topical", "exact_section")]
    hit_rate = sum(1 for r in topical_exact if r.get("retrieval_hit")) / len(topical_exact)
    citation_rate = sum(1 for r in topical_exact if r.get("citation_accurate")) / len(topical_exact)
    refusal_rows = [r for r in results if r["type"] == "refusal"]
    refusal_rate = sum(1 for r in refusal_rows if r["pass"]) / len(refusal_rows) if refusal_rows else None

    print(f"retrieval_hit_rate: {hit_rate:.2f}")
    print(f"citation_accuracy: {citation_rate:.2f}")
    if refusal_rate is not None:
        print(f"refusal_accuracy: {refusal_rate:.2f}")

    out_path = ROOT / "eval" / "results.json"
    out_path.write_text(
        json.dumps(
            {
                "results": results,
                "retrieval_hit_rate": hit_rate,
                "citation_accuracy": citation_rate,
                "refusal_accuracy": refusal_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
