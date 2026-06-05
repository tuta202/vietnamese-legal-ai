"""
hf_fetch_large.py — robust, resumable, chunked downloader for a single large
HF dataset file (the 3.5GB legacy/content.parquet), built for flaky links that
cut long-running connections.

Strategy: HTTP Range requests in fixed-size chunks appended to a .part file.
Resumes from the existing .part size; each chunk has its own timeout + retry,
so a dropped connection only costs the current chunk, not the whole download.
On success the .part is renamed to the final cache path (where hf_ingest reads).

CLI:
    python corpus/hf_fetch_large.py            # fetch legacy/content.parquet
    python corpus/hf_fetch_large.py --file data/content.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import json

import requests

_HERE = Path(__file__).resolve().parent
CACHE = _HERE / "data" / "hf_cache"
_CFG = json.loads((_HERE / "hf_config.json").read_text(encoding="utf-8"))
REPO = _CFG["source"]["repo"]
# Pinned commit for reproducibility (falls back to main if unset).
REVISION = _CFG.get("dataset_revision") or "main"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"

CHUNK = 16 * 1024 * 1024     # 16 MB per range request
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60


def _total_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=CONNECT_TIMEOUT)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def fetch(rel_path: str) -> Path:
    url = f"{BASE}/{rel_path}"
    final = CACHE / rel_path
    part = final.with_suffix(final.suffix + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)

    total = _total_size(url)
    have = part.stat().st_size if part.exists() else 0
    print(f"Target {rel_path}: {total/1e6:.1f} MB  (resuming from {have/1e6:.1f} MB)",
          flush=True)

    session = requests.Session()
    t0 = time.time()
    last_report = have
    while have < total:
        end = min(have + CHUNK - 1, total - 1)
        headers = {"Range": f"bytes={have}-{end}"}
        for attempt in range(1, 7):
            try:
                r = session.get(url, headers=headers, stream=True,
                                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(part, "ab") as f:
                    for block in r.iter_content(chunk_size=1024 * 256):
                        if block:
                            f.write(block)
                break
            except Exception as e:  # noqa: BLE001
                wait = min(2 ** attempt, 20)
                print(f"  chunk @{have/1e6:.0f}MB attempt {attempt}/6 failed: "
                      f"{e!r} — retry in {wait}s", flush=True)
                # truncate any partial bytes written past `have` so the next
                # Range request lines up exactly
                if part.exists() and part.stat().st_size > have:
                    with open(part, "r+b") as f:
                        f.truncate(have)
                time.sleep(wait)
        else:
            raise RuntimeError(f"giving up at offset {have}")
        have = part.stat().st_size
        if have - last_report >= 256 * 1024 * 1024:
            rate = (have) / max(time.time() - t0, 1) / 1e6
            print(f"  {have/1e6:.0f}/{total/1e6:.0f} MB  ({rate:.1f} MB/s)", flush=True)
            last_report = have

    if part.stat().st_size != total:
        raise RuntimeError(f"size mismatch: {part.stat().st_size} != {total}")
    part.replace(final)
    print(f"DONE {rel_path}: {total/1e6:.1f} MB in {time.time()-t0:.0f}s -> {final}",
          flush=True)
    return final


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="legacy/content.parquet")
    args = ap.parse_args()
    fetch(args.file)


if __name__ == "__main__":
    main()
