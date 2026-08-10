"""Cross-reference graph over Act sections (Graph RAG foundation).

Statutory text is naturally a graph: sections cite each other constantly
("shall have the meaning assigned to it in section 515(3)(b)", "subject
to the provisions of section 139"). Vector similarity is blind to these
edges - Section 172's text may be semantically unlike Section 511's even
though understanding 172 *requires* 511. This module extracts those edges
with a regex over the already-extracted section text (pure local compute,
no API calls) and persists an adjacency structure that Graph RAG uses to
expand retrieval one hop along real statutory references.

Edges are per-Act only: "section 139" inside a 1961-Act section means the
1961 Act's own section 139, never the 2025 Act's (an Act's internal
references are always to itself).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "processed" / "xref_graph.json"
CASE_GRAPH_PATH = ROOT / "data" / "processed" / "case_graph.json"

# "section 139", "sections 82,83,84", "section 515(3)(b)" - capture the
# bare section identifier (digits + optional letter suffix), ignore the
# sub-section parentheses. Also catch the "sections A, B and C" list form.
_REF_RE = re.compile(
    r"\bsections?\s+((?:\d{1,4}[A-Z]{0,4}(?:\([^)]{1,12}\))*(?:\s*,\s*|\s*(?:and|or|to)\s+)?)+)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d{1,4}(?:-?[A-Z]{1,4})?")
_PARENS_RE = re.compile(r"\([^)]*\)")


def _load_sections():
    for path in sorted((ROOT / "data" / "processed").glob("itact*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def build_graph():
    """Extract per-Act cross-reference edges and persist to GRAPH_PATH."""
    known = defaultdict(set)  # act_id -> set of real section numbers
    sections = list(_load_sections())
    for s in sections:
        known[s["act_id"]].add(s["section"])

    edges = defaultdict(list)  # "act_id::section" -> [referenced section numbers]
    for s in sections:
        refs = set()
        for m in _REF_RE.finditer(s["text"]):
            # strip parenthesized sub-section markers first - "511(1)"
            # must yield only "511", not a spurious reference to a
            # nonexistent "section 1" (verified live: without this, 149
            # sections appeared to cite section 1)
            cleaned = _PARENS_RE.sub(" ", m.group(1))
            for num in _NUM_RE.findall(cleaned):
                # only keep references that resolve to a real section of
                # the same Act - this drops years ("1961"), rule numbers,
                # and extraction noise for free
                if num != s["section"] and num in known[s["act_id"]]:
                    refs.add(num)
        if refs:
            edges[f"{s['act_id']}::{s['section']}"] = sorted(refs)

    GRAPH_PATH.write_text(json.dumps(edges, indent=1), encoding="utf-8")
    total_edges = sum(len(v) for v in edges.values())
    print(f"Graph: {len(edges)} sections with outgoing refs, {total_edges} edges -> {GRAPH_PATH}")
    return edges


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        if not GRAPH_PATH.exists():
            build_graph()
        _graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return _graph


def references_of(act_id, section):
    """Sections that (act_id, section) explicitly cites."""
    return get_graph().get(f"{act_id}::{section}", [])


def referenced_by(act_id, section):
    """Sections that cite (act_id, section) - reverse edges, computed on demand."""
    prefix = f"{act_id}::"
    return sorted(
        key[len(prefix):]
        for key, refs in get_graph().items()
        if key.startswith(prefix) and section in refs
    )


_case_graph = None


def cases_for(act_id, section):
    """Case names manually linked to (act_id, section) - see
    scripts/build_cases.py for how these links were verified. Kept in a
    separate file from xref_graph.json since cases aren't sections and
    have no outgoing statutory references of their own."""
    global _case_graph
    if _case_graph is None:
        _case_graph = json.loads(CASE_GRAPH_PATH.read_text(encoding="utf-8")) if CASE_GRAPH_PATH.exists() else {}
    return _case_graph.get(f"{act_id}::{section}", [])


if __name__ == "__main__":
    build_graph()
