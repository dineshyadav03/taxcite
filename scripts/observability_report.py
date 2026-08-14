"""Reads data/observability.jsonl and prints a usage/health report.

A report to run on demand or on a schedule, not real-time alerting - no
free equivalent of PagerDuty exists at this project's scale (see Phase 3
of the production-readiness plan). Computes real p50/p95/p99 latency per
pattern from actual logged requests, and flags two simple threshold-based
signals: a refusal-rate spike (possible faithfulness drift - the corpus
or a pattern started refusing much more than usual) and a token-usage
spike (possible cost anomaly - one pattern burning far more tokens per
request than its peers).
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from observability import read_log  # noqa: E402

# Thresholds are deliberately simple and stated, not tuned against real
# incident data (this project has none) - a refusal rate above 40% or a
# per-request token count more than 3x the overall average are flagged as
# worth a human look, not treated as proven problems.
REFUSAL_RATE_FLAG = 0.40
TOKEN_SPIKE_MULTIPLIER = 3.0


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    entries = read_log()
    if not entries:
        print("No observability data yet - data/observability.jsonl is empty or missing.")
        return

    by_pattern = defaultdict(list)
    for e in entries:
        by_pattern[e["pattern"]].append(e)

    total_cost = sum(e["estimated_cost_usd"] for e in entries)
    total_tokens = sum(e["prompt_tokens"] + e["completion_tokens"] for e in entries)
    overall_avg_tokens = total_tokens / len(entries)

    print(f"Total requests logged: {len(entries)}")
    print(f"Total estimated cost (Groq list price): ${total_cost:.6f}")
    print(f"Overall average tokens/request: {overall_avg_tokens:.0f}\n")

    flags = []
    for pattern, rows in sorted(by_pattern.items()):
        latencies = [r["latency_seconds"] for r in rows if not r["cache_hit"]]
        refused = sum(1 for r in rows if r["refused"])
        refusal_rate = refused / len(rows)
        cache_hits = sum(1 for r in rows if r["cache_hit"])
        avg_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / len(rows)

        print(f"[{pattern}] {len(rows)} requests, {cache_hits} cache hits")
        if latencies:
            print(
                f"  latency p50={percentile(latencies, 0.5):.2f}s "
                f"p95={percentile(latencies, 0.95):.2f}s "
                f"p99={percentile(latencies, 0.99):.2f}s"
            )
        else:
            print("  latency: no non-cached requests yet")
        print(f"  refusal_rate={refusal_rate:.2f}  avg_tokens/request={avg_tokens:.0f}")

        if refusal_rate > REFUSAL_RATE_FLAG:
            flags.append(f"[{pattern}] refusal rate {refusal_rate:.0%} exceeds {REFUSAL_RATE_FLAG:.0%} - possible drift")
        if avg_tokens > overall_avg_tokens * TOKEN_SPIKE_MULTIPLIER:
            flags.append(
                f"[{pattern}] avg tokens/request ({avg_tokens:.0f}) is "
                f">{TOKEN_SPIKE_MULTIPLIER}x the overall average ({overall_avg_tokens:.0f}) - possible cost anomaly"
            )

    print()
    if flags:
        print("FLAGGED:")
        for f in flags:
            print(f"  - {f}")
    else:
        print("No thresholds exceeded.")


if __name__ == "__main__":
    main()
