"""Pattern 11 - Jury RAG: run distinct patterns, treat agreement as confidence.

Every other pattern commits to one answer with no way to express
uncertainty. Jury RAG runs three mechanistically distinct existing
patterns on the same question - simple (exact/vector baseline),
corrective (relevance-graded retrieval), graph (statutory-dependency
expansion) - and takes each juror's top chunk (act_id, section) as its
vote. If 2 or 3 agree, that agreement is real signal: three different
retrieval strategies landed on the same provision. If none agree, that
disagreement is the finding, not a bug to hide - some questions genuinely
have more than one applicable reading, and picking one juror's answer
silently would hide that.

Sequential, not parallel: Voyage's free tier is 3 requests/minute, so
three concurrent embedding calls would just be rate-limit-queued anyway -
sequential costs nothing extra in wall clock.
"""
from collections import Counter

_JURORS = ["simple", "corrective", "graph"]


def answer(question, session_id=None):
    # avoid circular import: jury imports the registry lazily, same as adaptive.py
    from patterns import PATTERNS

    votes = []
    results = {}
    for name in _JURORS:
        result = PATTERNS[name](question, session_id=session_id)
        results[name] = result
        if result.get("refused") or not result.get("chunks"):
            vote = None
        else:
            top = result["chunks"][0]
            vote = (top["metadata"].get("act_id"), top["metadata"].get("section"))
        votes.append(
            {
                "pattern": name,
                "act": vote[0] if vote else None,
                "section": vote[1] if vote else None,
                "answer_preview": result["answer"][:200],
            }
        )

    valid_votes = [v for v in votes if v["section"] is not None]
    tally = Counter((v["act"], v["section"]) for v in valid_votes)
    winner, agreement_count = (tally.most_common(1)[0] if tally else (None, 0))

    trace = {
        "jurors": _JURORS,
        "votes": votes,
        "consensus": agreement_count >= 2,
        "agreement_count": agreement_count,
    }

    if agreement_count >= 2:
        winning_pattern = next(
            v["pattern"] for v in votes if (v["act"], v["section"]) == winner
        )
        winning_result = results[winning_pattern]
        trace["winning_juror"] = winning_pattern
        return {
            "answer": winning_result["answer"],
            "chunks": winning_result["chunks"],
            "refused": False,
            "trace": trace,
        }

    if not valid_votes:
        refusal = results[_JURORS[0]]
        return {
            "answer": refusal["answer"],
            "chunks": refusal.get("chunks", []),
            "refused": True,
            "trace": trace,
        }

    readings = "; ".join(
        f"{v['pattern']} points to {v['act']} Section {v['section']}" for v in valid_votes
    )
    disagreement_answer = (
        "Independent retrieval strategies disagree on the most relevant section "
        f"({readings}) - this question may have more than one applicable reading. "
        "See the individual juror answers below for each perspective.\n\n"
        + "\n\n".join(
            f"[{v['pattern']}] {results[v['pattern']]['answer']}"
            for v in votes
            if v["section"] is not None
        )
    )
    all_chunks = []
    seen_ids = set()
    for name in _JURORS:
        for c in results[name].get("chunks", []):
            if c["id"] not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c["id"])

    return {
        "answer": disagreement_answer,
        "chunks": all_chunks,
        "refused": False,
        "trace": trace,
    }
