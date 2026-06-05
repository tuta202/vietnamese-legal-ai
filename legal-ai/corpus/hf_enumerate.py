"""
hf_enumerate.py — enumerate real values of filter fields from the cached
metadata parquet files, so filter lists in hf_config.json are grounded in
actual data (not guessed). Reads ONLY the small metadata parquets.

CLI:
    python corpus/hf_enumerate.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
CACHE = _HERE / "data" / "hf_cache"

A_META = CACHE / "data" / "metadata.parquet"
B_META = CACHE / "legacy" / "metadata.parquet"


def _col_counter(path: Path, col: str) -> Counter:
    t = pq.read_table(path, columns=[col])
    return Counter(t.column(col).to_pylist())


def show(path: Path, col: str, top: int = 60) -> None:
    print(f"\n--- {path.name} :: {col} ---")
    if not path.exists():
        print("  (file not downloaded yet)")
        return
    c = _col_counter(path, col)
    print(f"  distinct={len(c)}  total={sum(c.values())}")
    for val, n in c.most_common(top):
        sval = str(val)
        if len(sval) > 70:
            sval = sval[:70] + "…"
        print(f"  {n:>8}  {sval}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("TIER A (data/metadata.parquet)")
    print("=" * 70)
    show(A_META, "loai_van_ban")
    show(A_META, "tinh_trang_hieu_luc")
    show(A_META, "linh_vuc", top=80)
    show(A_META, "nganh", top=40)

    print()
    print("=" * 70)
    print("TIER B (legacy/metadata.parquet)")
    print("=" * 70)
    show(B_META, "legal_type")
    show(B_META, "effect_status")
    show(B_META, "legal_sectors", top=80)


if __name__ == "__main__":
    main()
