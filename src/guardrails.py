"""Prompt-injection scanning on retrieved content - flag, don't block.

The corpus is a static official government PDF, not scraped or user-
uploaded content, so injection risk here is genuinely lower than a
typical RAG system's - but "lower risk" isn't "no risk," and this was a
real, unaddressed gap until now, not something safe by construction.

Flag-don't-block is a deliberate choice, not a placeholder: a false
positive that silently drops a real, relevant statute chunk would be a
worse failure for this project than a false negative that lets a flagged
chunk through - the citation-grounded design already means any
fabricated instruction embedded in retrieved text still has to compete
against "answer using only the retrieved text, cite your source" in the
system prompt, so blocking isn't the only defense layer here.
"""
import re

_INJECTION_PATTERNS = [
    re.compile(r"\bignore (all |the )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"\bnew instructions?\s*:", re.IGNORECASE),
    re.compile(r"\bdisregard (everything|all)\b", re.IGNORECASE),
    re.compile(r"\bact as (an?|the)\b.{0,30}\bassistant\b", re.IGNORECASE),
]


def scan_chunks(chunks):
    """Returns a list of {chunk_id, section, act, pattern} flags for any
    retrieved chunk whose text matches a suspicious injected-instruction
    pattern. Callers log/display this; nothing here removes or alters
    the chunk itself."""
    flags = []
    for c in chunks:
        text = c.get("text", "")
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                flags.append(
                    {
                        "chunk_id": c.get("id"),
                        "section": c.get("metadata", {}).get("section"),
                        "act": c.get("metadata", {}).get("act_id"),
                        "pattern": pattern.pattern,
                    }
                )
                break  # one flag per chunk is enough signal
    return flags
