"""Targeted re-verification of the two real bugs found in the first
verify_patterns.py run, plus a stronger graph-pattern test case."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from patterns import run  # noqa: E402

CASES = [
    ("memory", "What does section 139 of the Income-tax Act 2025 say?", "fix-verify-1"),
    ("self_rag", "Who can act as an authorised representative before income tax authorities?", None),
    ("graph", "What does section 139 of the Income-tax Act 1961 say and what does it depend on?", None),
]


def main():
    for i, (pattern, question, session) in enumerate(CASES):
        if i > 0:
            time.sleep(20)
        try:
            r = run(pattern, question, session_id=session)
            print(f"=== {pattern} ===")
            print(f"Q: {question}")
            print(f"refused={r['refused']}")
            print(f"A: {r['answer'][:350]}")
            acts_seen = sorted({c["metadata"]["act_id"] for c in r.get("chunks", [])})
            print(f"acts_in_sources: {acts_seen}")
            print(f"trace: {json.dumps(r['trace'], default=str)[:600]}")
        except Exception as e:
            print(f"=== {pattern} FAILED: {type(e).__name__}: {e} ===")
        print()


if __name__ == "__main__":
    main()
