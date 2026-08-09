"""Extract + chunk both Acts into data/processed/*.jsonl."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunk import dedupe_and_filter
from extract import extract_sections

ROOT = Path(__file__).resolve().parent.parent


def main():
    jobs = [
        ("itact2025", "Income-tax Act, 2025", ROOT / "data/raw/itact2025.pdf", True, 0),
        ("itact1961", "Income-tax Act, 1961", ROOT / "data/raw/itact1961.pdf", False, 29),
    ]
    for act_id, act_title, pdf_path, two_column, skip_pages in jobs:
        raw_sections = list(extract_sections(pdf_path, act_id, act_title, two_column, skip_pages))
        sections = dedupe_and_filter(raw_sections)
        sections.sort(key=lambda s: s["section"])
        out_path = ROOT / "data" / "processed" / f"{act_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for s in sections:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{act_id}: {len(raw_sections)} raw matches -> {len(sections)} sections -> {out_path}")


if __name__ == "__main__":
    main()
