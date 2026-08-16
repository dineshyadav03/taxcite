"""Reads data/feedback.jsonl and prints a per-pattern rating summary plus
every down-voted question, the actual point of a feedback loop - a
helpful-rate number alone doesn't tell anyone what to go fix, the list
of specific down-voted answers does.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feedback import read_log  # noqa: E402


def main():
    entries = read_log()
    if not entries:
        print("No feedback yet - data/feedback.jsonl is empty or missing.")
        return

    by_pattern = defaultdict(list)
    for e in entries:
        by_pattern[e["pattern"]].append(e)

    total_up = sum(1 for e in entries if e["rating"] == "up")
    total_down = sum(1 for e in entries if e["rating"] == "down")
    print(f"Total feedback: {len(entries)} ({total_up} up, {total_down} down)\n")

    for pattern, rows in sorted(by_pattern.items()):
        up = sum(1 for r in rows if r["rating"] == "up")
        down = sum(1 for r in rows if r["rating"] == "down")
        print(f"[{pattern}] {len(rows)} rated - {up} up, {down} down ({up / len(rows):.0%} helpful)")

    down_votes = [e for e in entries if e["rating"] == "down"]
    if down_votes:
        print("\nDOWN-VOTED ANSWERS (review these):")
        for e in down_votes:
            print(f"  [{e['pattern']}] {e['timestamp']}")
            print(f"    Q: {e['question']}")
            print(f"    A: {e['answer'][:200]}")
    else:
        print("\nNo down-voted answers.")


if __name__ == "__main__":
    main()
