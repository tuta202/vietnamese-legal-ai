"""
verify_vbpl_full_corpus.py
--------------------------
Verify effective status of ALL law_ids in corpus/data/corpus_clean.json
against vbpl.vn (as-of 2026-03-01).

Features:
- Loads all unique law_ids from corpus_clean.json
- Merges all existing caches (vbpl_effective_status*.jsonl) to skip already-verified
- Concurrent workers (default 4) with per-worker rate limiting
- Appends results to a single JSONL cache file (resumable)
- Writes a final summary CSV + JSON report

CLI:
    python verify_vbpl_full_corpus.py
    python verify_vbpl_full_corpus.py --workers 3 --sleep 1.5
    python verify_vbpl_full_corpus.py --dry-run          # show stats only, no HTTP
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VBPL_URL = "https://vbpl.vn/van-ban/trung-uong"
VBPL_SEARCH_ACTION = "c529d164f28418e5898a834422629e64c6816af1"
DEFAULT_AS_OF_DATE = "2026-03-01"

_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = _ROOT / "corpus" / "data" / "corpus_clean.json"
CACHE_PATH  = _ROOT / "cache" / "vbpl_full_corpus_verify.jsonl"
# Legacy caches to merge (read-only)
LEGACY_CACHES = [
    _ROOT / "cache" / "vbpl_effective_status.jsonl",
    _ROOT / "cache" / "vbpl_effective_status_full_corpus.jsonl",
    _ROOT / "cache" / "vbpl_effective_status_verified_toprisk.jsonl",
]
OUTPUT_JSON = _ROOT / "outputs" / "vbpl_full_corpus_status.json"
OUTPUT_CSV  = _ROOT / "outputs" / "vbpl_full_corpus_status.csv"

log = logging.getLogger("verify_full")

# ---------------------------------------------------------------------------
# Effective-status normalization helpers.
# ---------------------------------------------------------------------------

def norm_id(value: str) -> str:
    """Normalise a law_id for dedup keying."""
    value = " ".join(str(value or "").strip().split())
    value = value.strip("'\"""''")
    return value.upper().replace("Ð", "Đ")


def parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            pass
    return None


def class_as_of(record: dict | None, as_of: date) -> tuple[str, str]:
    if not record:
        return "vbpl_not_found", "No exact VBPL record found."
    eff_from = parse_iso_date(record.get("effFrom"))
    eff_to   = parse_iso_date(record.get("effTo"))
    status   = ((record.get("effStatus") or {}).get("name") or "").strip().lower()
    if eff_from and eff_from > as_of:
        return "not_yet_effective_as_of", f"effFrom={eff_from.isoformat()} is after as_of={as_of}."
    if eff_to and eff_to <= as_of:
        return "expired_before_as_of", f"effTo={eff_to.isoformat()} is on/before as_of={as_of}."
    if "hết hiệu lực một phần" in status or "ngưng hiệu lực một phần" in status:
        return "partial_effective_as_of", "Partial effect status on VBPL."
    if "hết hiệu lực toàn bộ" in status and not eff_to:
        return "expired_full_unknown_date", "VBPL status=expired_full but effTo missing."
    if "chưa có hiệu lực" in status:
        return "not_yet_effective_unknown_date", "VBPL status=not_yet_effective but effFrom missing/unparseable."
    if record:
        return "valid_as_of", "No date/status invalidated this document at as_of."
    return "unknown", "Unhandled."


def repair_mojibake(value: Any) -> Any:
    if isinstance(value, str):
        if any(marker in value for marker in ("Ã", "áº", "á»", "Ä", "Æ")):
            try:
                return value.encode("latin1").decode("utf-8")
            except UnicodeError:
                return value
        return value
    if isinstance(value, list):
        return [repair_mojibake(item) for item in value]
    if isinstance(value, dict):
        return {k: repair_mojibake(v) for k, v in value.items()}
    return value


def parse_rsc_json(text: str) -> dict:
    match = re.search(r'1:([\{\[].*)', text)
    if not match:
        for line in text.splitlines():
            if line.startswith("1:"):
                return json.loads(line[2:])
        raise ValueError("No '1:' line in RSC response")

    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, idx = decoder.raw_decode(json_str)
        return obj


def vbpl_search(session: requests.Session, law_id: str, page_size: int = 10) -> dict:
    payload = [{"pageNumber": 1, "pageSize": page_size, "keyword": law_id, "optionDoc": "title"}]
    headers = {
        "Next-Action": VBPL_SEARCH_ACTION,
        "Content-Type": "text/plain;charset=UTF-8",
        "Accept": "text/x-component",
        "Origin": "https://vbpl.vn",
        "Referer": VBPL_URL,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    resp = session.post(VBPL_URL, headers=headers, data=body, timeout=45)
    resp.raise_for_status()
    text = resp.content.decode("utf-8", errors="replace")
    return repair_mojibake(parse_rsc_json(text))


def extract_year(law_id: str) -> int | None:
    match = re.search(r'/((?:19|20)\d{2})/', law_id)
    if match:
        return int(match.group(1))
    return None


def verify_one(session: requests.Session, law_id: str, as_of: date) -> dict:
    payload = vbpl_search(session, law_id)
    items = payload.get("items") or []
    exact = [item for item in items if norm_id(item.get("docNum", "")) == norm_id(law_id)]
    orig_exact_count = len(exact)

    # Filter out English translations if possible
    non_translation = [
        item for item in exact
        if (item.get("docType") or {}).get("code") != "BD"
        and "bản dịch" not in ((item.get("docType") or {}).get("name") or "").lower()
    ]
    if non_translation:
        exact = non_translation

    # Match by year if still multiple matches
    year = extract_year(law_id)
    if len(exact) > 1 and year:
        by_year = []
        for item in exact:
            issue_date = parse_iso_date(item.get("issueDate"))
            if issue_date and issue_date.year == year:
                by_year.append(item)
        if by_year:
            exact = by_year

    picked = exact[0] if exact else None
    cls, reason = class_as_of(picked, as_of)
    eff_from_str = (picked or {}).get("effFrom", "")
    eff_to_str   = (picked or {}).get("effTo", "")
    eff_status   = ((picked or {}).get("effStatus") or {}).get("name", "") if picked else ""
    return {
        "law_id":                   law_id,
        "source":                   "vbpl.vn",
        "effective_class_as_of":    cls,
        "effective_class_reason":   reason,
        "effFrom":                  eff_from_str,
        "effTo":                    eff_to_str,
        "effStatus":                eff_status,
        "exact_match_count":        orig_exact_count,
        "smart_filtered":           True,
        "total_results":            payload.get("total"),
        "candidate_doc_nums":       [item.get("docNum") for item in items[:5]],
        "record":                   picked,
    }

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_all_caches() -> dict[str, dict]:
    """Merge legacy caches + current cache. Returns {norm_id: row}."""
    merged: dict[str, dict] = {}
    all_paths = LEGACY_CACHES + [CACHE_PATH]
    for path in all_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("effective_class_as_of") == "error":
                        continue
                    if row.get("exact_match_count", 0) > 1 and not row.get("smart_filtered"):
                        # Skip multi-matches to force re-verification with new smart filter
                        continue
                    key = norm_id(row.get("law_id", ""))
                    if key and key not in merged:
                        merged[key] = row
                except json.JSONDecodeError:
                    pass
    return merged


_append_lock = threading.Lock()

def append_cache(row: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _append_lock:
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def load_law_ids_from_corpus() -> list[str]:
    """Extract sorted unique law_ids from corpus_clean.json."""
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    arts = data.get("articles", data) if isinstance(data, dict) else data
    ids = sorted(set(
        str(a["law_id"]).strip()
        for a in arts
        if a.get("law_id") and str(a["law_id"]).strip()
    ))
    return ids

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(
    law_id: str,
    as_of: date,
    sleep_secs: float,
    max_retries: int,
) -> dict:
    """Verify a single law_id. Creates its own session (thread-safe)."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    # Warm-up cookie
    try:
        session.get(VBPL_URL, timeout=30)
    except Exception:
        pass

    for attempt in range(max_retries):
        try:
            row = verify_one(session, law_id, as_of)
            time.sleep(sleep_secs)
            return row
        except Exception as exc:
            if attempt == max_retries - 1:
                return {"law_id": law_id, "source": "vbpl.vn", "error": repr(exc),
                        "effective_class_as_of": "error"}
            wait = sleep_secs * (2 ** attempt)
            log.warning("Retry %d/%d for %s after %.1fs: %s", attempt + 1, max_retries, law_id, wait, exc)
            time.sleep(wait)
    # Should never reach here
    return {"law_id": law_id, "source": "vbpl.vn", "error": "max_retries exceeded",
            "effective_class_as_of": "error"}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Verify ALL corpus law_ids against vbpl.vn effective status."
    )
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE,
                        help="Cutoff date YYYY-MM-DD (default: 2026-03-01)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent HTTP workers (default: 4)")
    parser.add_argument("--sleep", type=float, default=1.2,
                        help="Sleep seconds between requests per worker (default: 1.2)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="Max retries per law_id (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats and exit without any HTTP calls")
    parser.add_argument("--reset-cache", action="store_true",
                        help="Ignore existing cache (re-verify everything)")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()

    # ---- Load corpus law_ids ----
    log.info("Loading law_ids from %s …", CORPUS_PATH)
    all_ids = load_law_ids_from_corpus()
    log.info("  %d unique law_ids found", len(all_ids))

    # ---- Load existing cache ----
    if args.reset_cache:
        cache: dict[str, dict] = {}
        log.info("  --reset-cache: ignoring all caches")
    else:
        cache = load_all_caches()
        log.info("  %d already cached (will skip)", len(cache))

    todo = [lid for lid in all_ids if norm_id(lid) not in cache]
    log.info("  %d remaining to verify", len(todo))

    if args.dry_run:
        print(f"\n[DRY RUN] would verify {len(todo)} law_ids with {args.workers} workers")
        print(f"  as_of: {as_of}, sleep: {args.sleep}s, max_retries: {args.max_retries}")
        print(f"  estimated time: ~{len(todo) * args.sleep / args.workers / 60:.1f} minutes")
        return

    # ---- Run parallel verification ----
    results_map: dict[str, dict] = dict(cache)  # seed with cached

    total = len(todo)
    done_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_id = {
            pool.submit(_worker, lid, as_of, args.sleep, args.max_retries): lid
            for lid in todo
        }
        for future in as_completed(future_to_id):
            lid = future_to_id[future]
            done_count += 1
            try:
                row = future.result()
            except Exception as exc:
                row = {"law_id": lid, "source": "vbpl.vn", "error": repr(exc),
                       "effective_class_as_of": "error"}

            append_cache(row)
            results_map[norm_id(lid)] = row

            elapsed = time.time() - start_time
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total - done_count) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %s → %s  (%.1f/min, ETA %.0fmin)",
                done_count, total, lid,
                row.get("effective_class_as_of", "?"),
                rate * 60, eta / 60,
            )

    # ---- Merge remaining cached entries not in todo ----
    for lid in all_ids:
        key = norm_id(lid)
        if key not in results_map:
            results_map[key] = {"law_id": lid, "effective_class_as_of": "missing"}

    # ---- Build final ordered list (same order as all_ids) ----
    final_rows = [results_map.get(norm_id(lid), {"law_id": lid}) for lid in all_ids]

    # ---- Write JSON ----
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d rows → %s", len(final_rows), OUTPUT_JSON)

    # ---- Write CSV summary ----
    csv_fields = [
        "law_id", "effective_class_as_of", "effective_class_reason",
        "effFrom", "effTo", "effStatus", "exact_match_count", "total_results",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in final_rows:
            writer.writerow({k: row.get(k, "") for k in csv_fields})
    log.info("Wrote CSV → %s", OUTPUT_CSV)

    # ---- Print summary ----
    from collections import Counter
    class_counts = Counter(r.get("effective_class_as_of", "?") for r in final_rows)
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"  as_of: {as_of}  |  total law_ids: {len(final_rows)}")
    print(f"  {'class':<40} {'count':>6}")
    print(f"  {'-'*48}")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        marker = " ← INVALID" if cls in ("expired_before_as_of", "not_yet_effective_as_of",
                                           "expired_full_unknown_date") else ""
        print(f"  {cls:<40} {cnt:>6}{marker}")

    invalid_classes = {"expired_before_as_of", "not_yet_effective_as_of", "expired_full_unknown_date"}
    invalid_count = sum(1 for r in final_rows if r.get("effective_class_as_of") in invalid_classes)
    print(f"\n  => {invalid_count} law_ids INVALID at {as_of} (should be filtered from corpus)")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
