"""Extract tables from the Act PDFs as a second modality (Multi-modal RAG).

The prose extraction pipeline (src/extract.py) deliberately reads tables
as garbled text - rate schedules and Schedule tables were a documented,
verified failure mode (the 2025 Act's Section 194 rate table caused
degenerate LLM repetition). This script treats tables as their own
modality: pdfplumber's table detector pulls each table's cell grid, rows
are serialized as pipe-delimited lines (structure preserved instead of
interleaved prose), and each table becomes one retrievable chunk tagged
modality="table" with the page it came from and the nearest preceding
bold section number for citation.

"Multi-modal" here means text + structured tables - the honest version of
the pattern for a statute corpus, which contains no images or audio worth
indexing.
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "processed" / "tables.jsonl"

_SECTION_NO_RE = re.compile(r"^(\d{1,4}[A-Z]{0,4})\.?$")


def serialize_table(rows):
    """Pipe-delimited rows, empty cells normalized, blank rows dropped."""
    lines = []
    for row in rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_tables(pdf_path, act_id, act_title):
    """Yield table chunks with page + nearest-preceding-section context."""
    current_section = None
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # track the last bold section number seen so tables can be
            # attributed to the section they appear under (same bold-font
            # signal extract.py's boundary detection is built on)
            for w in page.extract_words(extra_attrs=["fontname"]):
                if "bold" in w["fontname"].lower() and _SECTION_NO_RE.match(w["text"]):
                    current_section = _SECTION_NO_RE.match(w["text"]).group(1)
            for t_idx, table in enumerate(page.extract_tables()):
                # noise gate: a real table has at least 2 rows and 2 columns
                if len(table) < 2 or max(len(r) for r in table) < 2:
                    continue
                text = serialize_table(table)
                if len(text) < 40:
                    continue
                yield {
                    "act_id": act_id,
                    "act_title": act_title,
                    "section": current_section,
                    "page": page_num,
                    "table_index": t_idx,
                    "modality": "table",
                    "text": text,
                }


def main():
    jobs = [
        ("itact2025", "Income-tax Act, 2025", ROOT / "data/raw/itact2025.pdf"),
        ("itact1961", "Income-tax Act, 1961", ROOT / "data/raw/itact1961.pdf"),
    ]
    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for act_id, act_title, pdf_path in jobs:
            act_count = 0
            for chunk in extract_tables(pdf_path, act_id, act_title):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                act_count += 1
            print(f"{act_id}: {act_count} tables")
            count += act_count
    print(f"Total {count} table chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
