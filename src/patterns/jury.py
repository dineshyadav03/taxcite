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

All three jurors' *first* retrieval call embeds the exact same question
text, so the query vector is identical across them - computed once here
and passed to each juror instead of three separate Voyage calls (four,
counting Corrective's possible retry). Verified live: this was the
actual cause of Jury RAG's 100-180s latency, not generation time - three
sequential embedding calls under Voyage's 3 RPM free-tier throttle means
the 2nd and 3rd calls routinely queue behind a 21-42s backoff. One shared
embedding removes two of those three calls outright.
"""
from collections import Counter

from retrieve import embed_query

_JURORS = ["simple", "corrective", "graph"]


def answer(question, session_id=None):
    # avoid circular import: jury imports the registry lazily, same as adaptive.py
    from patterns import PATTERNS

    embedding = embed_query(question)

    votes = []
    results = {}
    for name in _JURORS:
        result = PATTERNS[name](question, session_id=session_id, embedding=embedding)
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
    refusal_count = len(_JURORS) - len(valid_votes)
    tally = Counter((v["act"], v["section"]) for v in valid_votes)
    winner, agreement_count = (tally.most_common(1)[0] if tally else (None, 0))

    trace = {
        "jurors": _JURORS,
        "votes": votes,
        "consensus": agreement_count >= 2,
        "agreement_count": agreement_count,
        "refusal_count": refusal_count,
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

    if refusal_count >= 2:
        # A majority of jurors declining to answer is itself a majority
        # signal, symmetric to 2-of-3 agreeing being treated as consensus
        # - verified live via the golden-set eval: without this, a single
        # juror's low-confidence outlier vote (1 of 3, with the other 2
        # correctly refusing) was surfacing as a "disagreement" finding
        # instead of an overall refusal, on a genuinely out-of-corpus
        # question (a GST rate question - GST is a separate Act this
        # corpus doesn't cover). 2-or-3-of-3 refusing now refuses overall.
        refusing_pattern = next(name for name, v in zip(_JURORS, votes) if v["section"] is None)
        refusal = results[refusing_pattern]
        trace["majority_refused"] = True
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
