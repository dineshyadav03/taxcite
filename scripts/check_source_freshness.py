"""Re-fetches the source PDFs and compares their hash against
data/processed/BUILD_MANIFEST.json, flagging if the government source
has changed since this project's index was last built - a Finance Act
amendment could update either Act's PDF without this project noticing
otherwise. Detection only; a real change means re-running the build
pipeline (scripts/build_chunks.py onward), not something this script
does on its own.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_sources import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    manifest_path = ROOT / "data" / "processed" / "BUILD_MANIFEST.json"
    if not manifest_path.exists():
        print("No BUILD_MANIFEST.json found - run scripts/build_manifest.py first.")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Manifest built at: {manifest['built_at']}")

    drift_found = False
    for src in manifest["sources"]:
        print(f"Checking {src['id']}...")
        resp = fetch(src["url"])
        live_hash = hashlib.sha256(resp.content).hexdigest()
        if live_hash == src["sha256"]:
            print(f"  up to date (sha256 matches)")
        else:
            drift_found = True
            print(f"  DRIFT DETECTED: source has changed since last indexed")
            print(f"    manifest sha256: {src['sha256']}")
            print(f"    live sha256:     {live_hash}")

    if drift_found:
        print("\nAt least one source has changed - consider re-running the build pipeline.")
        return 1
    print("\nAll sources match the last build. Corpus is current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
