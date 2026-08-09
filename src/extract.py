"""Extract section-level chunks directly from the source Act PDFs.

Both Acts typeset the section number (and, for the 1961 Act, the section
title too) in a bold font distinct from the surrounding body text - and
critically, unrelated numbered content (Schedule table rows, list items)
is set in the regular weight. That's a far more reliable section-boundary
signal than pattern-matching flattened text, which false-positives on any
number that happens to land at the start of a wrapped line (verified: a
plain regex pass produced 5,471 "matches" for the 1961 Act alone, when the
true count is nowhere close).

itact2025.pdf also uses the classic Gazette-of-India two-column layout: a
wide main-text column plus a narrow marginal-note caption column whose
side (left/right) alternates by page. We drop the margin column entirely
(same column-split logic as before) since it can't be reliably
re-associated with its section without much more work than a v1 warrants.
"""
import re

import pdfplumber

_HEADER_BAND_TOP = 50  # running header/page-number line sits above this
_SECTION_NO_RE = re.compile(r"^(\d{1,4}[A-Z]{0,4}(?:-[A-Z0-9]{1,4})?)\.?$")
_CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+[A-Z]?$")
_LINE_GROUP_TOL = 2.5


def _main_column_words(words, min_gap=15, margin_word_ceiling=12):
    """Drop the marginal-note column, keeping the main body text column."""
    if not words:
        return words
    body = [w for w in words if w["top"] >= _HEADER_BAND_TOP]
    if not body:
        return words
    excluded_ids = set()
    while True:
        remaining = [w for w in body if id(w) not in excluded_ids]
        xs = sorted(set(round(w["x0"], 1) for w in remaining))
        if len(xs) < 2:
            break
        gaps = sorted(
            ((b - a, (a + b) / 2) for a, b in zip(xs, xs[1:]) if b - a >= min_gap),
            reverse=True,
        )
        if not gaps:
            break
        _, split = gaps[0]
        left = [w for w in remaining if w["x0"] < split]
        right = [w for w in remaining if w["x0"] >= split]
        if not left or not right:
            break
        minority = left if len(left) <= len(right) else right
        if len(minority) > margin_word_ceiling:
            break
        excluded_ids.update(id(w) for w in minority)
    return [w for w in words if id(w) not in excluded_ids]


def _group_lines(words):
    """Group words into visual lines by top position, sorted reading-order."""
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= _LINE_GROUP_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def _is_bold(fontname):
    return "bold" in fontname.lower()


def _page_lines(page, two_column):
    words = page.extract_words(extra_attrs=["fontname"])
    if two_column:
        words = _main_column_words(words)
    else:
        words = [w for w in words if w["top"] >= _HEADER_BAND_TOP]
    return _group_lines(words)


def extract_sections(path, act_id, act_title, two_column, skip_pages=0):
    """Walk the PDF page by page, yielding section dicts in document order."""
    current_chapter = None
    current = None  # in-progress section dict

    def flush():
        if current and len("".join(current["_parts"])) >= 20:
            text = " ".join(p.strip() for p in current["_parts"] if p.strip())
            # the section-number token and its trailing "." are split into
            # separate words by the PDF (bold vs regular font runs), so the
            # body can start with a stray leading "." - strip it before
            # prepending our own "N. " so citations don't read "N. . (1)"
            text = text.lstrip(". ")
            yield {
                "act_id": act_id,
                "act_title": act_title,
                "chapter": current["chapter"],
                "section": current["section"],
                "title": current["title"],
                "text": f"{current['section']}. {text}",
            }

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[skip_pages:]:
            for line in _page_lines(page, two_column):
                # The page-level margin-note split can still leave a short
                # non-bold caption glued to the front of a section-start
                # line (both happened to land in the "main" cluster for
                # that page). Fall back to a per-line scan: if a bold
                # section-number token appears within the first few words
                # and everything before it is short and non-bold, treat
                # that token as the true line start.
                if not (_SECTION_NO_RE.match(line[0]["text"]) and _is_bold(line[0]["fontname"])):
                    for idx, w in enumerate(line[1:7], start=1):
                        if _SECTION_NO_RE.match(w["text"]) and _is_bold(w["fontname"]):
                            lead = line[:idx]
                            if not any(_is_bold(lw["fontname"]) for lw in lead) and len(
                                " ".join(lw["text"] for lw in lead)
                            ) <= 30:
                                line = line[idx:]
                            break

                first = line[0]
                line_text = " ".join(w["text"] for w in line)

                if _CHAPTER_RE.match(line_text.strip()) and _is_bold(first["fontname"]):
                    current_chapter = line_text.strip()
                    continue

                m = _SECTION_NO_RE.match(first["text"])
                if m and _is_bold(first["fontname"]):
                    yield from flush()
                    rest_words = line[1:]
                    title_words, body_words = [], []
                    in_title = True
                    for w in rest_words:
                        if in_title and _is_bold(w["fontname"]):
                            title_words.append(w["text"])
                        else:
                            in_title = False
                            body_words.append(w["text"])
                    title = " ".join(title_words).strip().rstrip(".—–-").strip() or None
                    body = " ".join(body_words)
                    current = {
                        "chapter": current_chapter,
                        "section": m.group(1),
                        "title": title,
                        "_parts": [body] if body else [],
                    }
                    continue

                if current is not None:
                    current["_parts"].append(line_text)
    yield from flush()
