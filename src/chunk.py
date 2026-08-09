"""Post-filtering for extracted sections (boundary detection lives in extract.py)."""


def dedupe_and_filter(sections, min_len=20):
    """Drop too-short fragments and collapse duplicate section numbers by
    keeping the longest text seen for that number. No max-length cap:
    heavily-amended sections (e.g. 1961 Act section 139, "Return of
    income", carries 60+ years of Finance Act amendments) can legitimately
    run 40k+ characters, and two earlier size-cap attempts (8000, then
    40000) each silently dropped real sections before this was caught by
    manually checking specific missing sections against the source PDF -
    a fixed ceiling on legal text length just isn't a sound heuristic.
    Splitting long sections into embedding-sized pieces is a separate,
    later concern - handled at index time, not here, so
    data/processed/*.jsonl stays one clean row per real Act section."""
    best = {}
    for s in sections:
        if len(s["text"]) < min_len:
            continue
        key = s["section"]
        if key not in best or len(s["text"]) > len(best[key]["text"]):
            best[key] = s
    return list(best.values())
