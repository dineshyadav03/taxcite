"""Writes data/processed/BUILD_MANIFEST.json - a snapshot of what the
current index was built from, so a later run can detect drift instead
of silently trusting a build that might be stale.

Rollback itself doesn't need new tooling here: chroma_db/ is committed
to git (see README's Deployment section), so `git checkout <sha> --
chroma_db/` already recovers any prior build. This script is about
*detecting* that a rebuild might be needed, not providing the rollback.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(), "sources": []}

    for src in sources:
        if src.get("status") == "deferred":
            continue
        pdf_path = ROOT / src["local_path"]
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping in manifest")
            continue
        manifest["sources"].append(
            {
                "id": src["id"],
                "url": src["url"],
                "local_path": src["local_path"],
                "sha256": _sha256(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
            }
        )

    for name in ("itact1961.jsonl", "itact2025.jsonl", "tables.jsonl"):
        p = ROOT / "data" / "processed" / name
        if p.exists():
            manifest.setdefault("chunk_counts", {})[name] = sum(1 for _ in p.open(encoding="utf-8"))

    out_path = ROOT / "data" / "processed" / "BUILD_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
