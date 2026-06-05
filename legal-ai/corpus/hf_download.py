"""
hf_download.py — resumable, cached download of the th1nhng0 legal-documents
parquet files into corpus/data/hf_cache/.

Uses huggingface_hub.hf_hub_download (NOT datasets.load_dataset) because:
  1. hf_hub_download resumes partial downloads on flaky networks.
  2. datasets streaming throws ArrowInvalid 'large_string -> string' on the
     oversized content columns; we read the parquet directly with pyarrow later.

CLI:
    python corpus/hf_download.py            # download all 4 files (resume if partial)
    python corpus/hf_download.py --small    # only the 2 metadata + tier-A content
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import json

from huggingface_hub import hf_hub_download

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "data" / "hf_cache"

_CFG = json.loads((_HERE / "hf_config.json").read_text(encoding="utf-8"))
REPO = _CFG["source"]["repo"]
# Pinned commit for reproducibility (falls back to main if unset).
REVISION = _CFG.get("dataset_revision") or "main"

# (filename, approx_MB, tier-label)
FILES = [
    ("data/metadata.parquet",   14,   "A-meta"),
    ("legacy/metadata.parquet", 51,   "B-meta"),
    ("data/content.parquet",    412,  "A-content"),
    ("legacy/content.parquet",  3508, "B-content"),
]


def download_one(filename: str, label: str, approx_mb: int) -> Path:
    t0 = time.time()
    print(f"[{label}] downloading {filename} (~{approx_mb} MB)…", flush=True)
    for attempt in range(1, 6):
        try:
            path = hf_hub_download(
                repo_id=REPO,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                local_dir=str(CACHE_DIR),
            )
            dt = time.time() - t0
            sz = Path(path).stat().st_size / 1e6
            print(f"[{label}] OK {sz:.1f} MB in {dt:.0f}s -> {path}", flush=True)
            return Path(path)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] attempt {attempt}/5 failed: {e!r}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"[{label}] giving up after 5 attempts")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true",
                    help="skip the 3.5GB legacy/content.parquet")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in FILES if not (args.small and f[0] == "legacy/content.parquet")]

    print(f"Cache dir: {CACHE_DIR}")
    print(f"Downloading {len(files)} file(s)…\n")
    for filename, approx_mb, label in files:
        download_one(filename, label, approx_mb)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
