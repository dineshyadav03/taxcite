"""Evaluate Jury RAG against the standard golden set, plus a metric that
actually matters for Jury's value proposition: how often its three
jurors (Simple, Corrective, Graph) reach consensus versus genuinely
disagree. Citation accuracy alone doesn't capture that - two patterns
could both cite the right section 100% of the time while one reaches
that answer through unanimous agreement and the other through the
"disagreement" fallback path, which is a materially different claim
about how much to trust the answer.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from patterns import run as run_pattern  # noqa: E402
from run_eval import check_citation_accuracy, check_retrieval_hit, load_golden  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run():
    golden = load_golden()
    results = []

    for item in golden:
        qtype = item["query_type"]
        row = {"id": item["id"], "question": item["question"], "type": qtype}
        result = run_pattern("jury", item["question"])
        trace = result.get("trace", {})
        row["consensus"] = trace.get("consensus")
        row["agreement_count"] = trace.get("agreement_count")

        if qtype == "refusal":
            row["refused"] = result["refused"]
            row["pass"] = result["refused"]
        elif qtype == "comparison":
            row["refused"] = result["refused"]
            row["pass"] = not result["refused"]
        else:
            hit = check_retrieval_hit(item, result.get("chunks", []))
            cited = check_citation_accuracy(item, result)
            row["retrieval_hit"] = hit
            row["citation_accurate"] = cited
            row["refused"] = result["refused"]
            row["pass"] = hit and (cited is True)

        results.append(row)
        print(f"[{'PASS' if row['pass'] else 'FAIL'}] {item['id']}: consensus={row['consensus']} agreement={row['agreement_count']}")

    scoreable = [r for r in results if r["type"] not in ("comparison",)]
    passed = sum(1 for r in scoreable if r["pass"])
    print(f"\njury: {passed}/{len(scoreable)} passed ({passed / len(scoreable) * 100:.0f}%)")

    scored_for_consensus = [r for r in results if r["consensus"] is not None]
    consensus_rate = sum(1 for r in scored_for_consensus if r["consensus"]) / len(scored_for_consensus)
    print(f"consensus_rate: {consensus_rate:.2f} ({sum(1 for r in scored_for_consensus if r['consensus'])}/{len(scored_for_consensus)})")

    out_path = ROOT / "eval" / "results_jury.json"
    out_path.write_text(json.dumps({"results": results, "consensus_rate": consensus_rate}, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
