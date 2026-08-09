"""Download the source PDFs listed in data/sources.json into data/raw/.

Government/institutional PDF hosts here (ICAI's CloudFront, India Code) have
shown transient TLS handshake resets under plain requests - retry with
backoff rather than failing on the first blip.
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MAX_ATTEMPTS = 4


def fetch(url):
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = 2**attempt
            print(f"  attempt {attempt} failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise last_err


def main():
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    for src in sources:
        if src.get("status") == "deferred":
            print(f"Skipping {src['title']} (deferred: {src['deferred_reason']})")
            continue
        dest = ROOT / src["local_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {src['title']} -> {dest}")
        resp = fetch(src["url"])
        if resp.headers.get("Content-Type", "").split(";")[0] != "application/pdf":
            print(f"  WARNING: unexpected Content-Type {resp.headers.get('Content-Type')!r}", file=sys.stderr)
        dest.write_bytes(resp.content)
        print(f"  OK, {len(resp.content):,} bytes")


if __name__ == "__main__":
    main()
