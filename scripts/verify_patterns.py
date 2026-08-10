"""Run every RAG pattern against a question chosen to exercise its
specific mechanism, and print answer + trace summary for manual review.

Sequential with ~20s spacing between patterns: each pattern spends 1-3
Voyage embedding calls and the free tier is 3 requests/minute - running
back-to-back would just convert the whole run into 429-retry loops.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from patterns import run  # noqa: E402

CASES = [
    ("simple", "What does section 271A of the Income-tax Act, 1961 say?", None),
    ("memory", "What does section 139 of the Income-tax Act 2025 say?", "verify-session"),
    # follow-up in the same session - only answerable via memory
    ("memory", "And what about under the old Act?", "verify-session"),
    ("branched", "How does the definition of agricultural income compare between the 1961 Act and the 2025 Act?", None),
    ("hyde", "Do I have to pay tax on money my friend transferred to me as a gift?", None),
    ("adaptive", "What is the penalty for not maintaining books of account?", None),
    ("corrective", "What happens if someone files their income tax return late?", None),
    ("self_rag", "Who can act as an authorised representative before income tax authorities?", None),
    ("agentic", "What conditions in section 139 of the 1961 Act refer to other sections, and what do those sections cover?", None),
    ("multimodal", "What rate of income-tax applies to income from Global Depository Receipts?", None),
    ("graph", "What does section 115VI of the 1961 Act say about relevant shipping income?", None),
]


def summarize_trace(trace, limit=500):
    s = json.dumps(trace, default=str)
    return s[:limit] + ("..." if len(s) > limit else "")


def main():
    results = []
    for i, (pattern, question, session) in enumerate(CASES):
        if i > 0:
            time.sleep(20)
        t0 = time.perf_counter()
        try:
            r = run(pattern, question, session_id=session)
            elapsed = time.perf_counter() - t0
            print(f"=== {pattern} ({elapsed:.0f}s) ===")
            print(f"Q: {question}")
            print(f"refused={r['refused']}  resolved_pattern={r.get('pattern')}")
            print(f"A: {r['answer'][:400]}")
            print(f"trace: {summarize_trace(r['trace'])}")
            sources = {(c['metadata'].get('act_id'), c['metadata'].get('section'), c['metadata'].get('modality', 'prose')) for c in r.get('chunks', [])}
            print(f"sources: {sorted(str(s) for s in sources)[:8]}")
            results.append((pattern, "OK" if not r["refused"] else "REFUSED"))
        except Exception as e:
            print(f"=== {pattern} FAILED ===\n{type(e).__name__}: {e}")
            results.append((pattern, f"ERROR: {type(e).__name__}"))
        print()

    print("=" * 50)
    for pattern, status in results:
        print(f"{pattern:12s} {status}")


if __name__ == "__main__":
    main()
