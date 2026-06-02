"""
Crawler for https://vbpl.vn — primary source for Vietnamese legal documents.
Falls back to thuvien_crawler if a document cannot be found here.

NOTE: vbpl.vn was fully redesigned as a SPA in 2024. All legacy SharePoint paths
(/TW/Pages/...) now return 404. crawl() returns None immediately so the
thuvienphapluat.vn fallback is used without wasting retries.
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "crawler" / "crawler_config.json"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def safe_id(law_id: str) -> str:
    return re.sub(r"[/ ]", "_", law_id)


def fetch_with_retry(url: str, params: dict | None = None, retries: int = 3, timeout: int = 20) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            log.warning("Timeout attempt %d/%d for %s", attempt, retries, url)
            if attempt < retries:
                time.sleep(2)
        except requests.exceptions.HTTPError as exc:
            # 4xx/5xx are definitive — no point retrying
            log.warning("HTTP %s for %s — skipping retries", exc.response.status_code, url)
            return None
        except requests.exceptions.RequestException as exc:
            log.warning("Request error attempt %d/%d for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2)
    return None


def crawl(law_entry: dict) -> str | None:
    """
    Attempt to fetch full text from vbpl.vn.

    vbpl.vn migrated to a SPA in 2024 — legacy HTML endpoints all return 404.
    Returns None so process_law() proceeds to the thuvienphapluat.vn fallback.
    This stub is kept so a future vbpl.vn API implementation can slot in here
    without touching process_law().
    """
    log.debug("vbpl.vn: unavailable (SPA migration) for %s", law_entry["id"])
    return None


def process_law(law_entry: dict) -> bool:
    """Download and save one law. Returns True on success."""
    sid = safe_id(law_entry["id"])
    out_path = OUTPUT_DIR / f"{sid}.txt"
    todo_path = OUTPUT_DIR / f"TODO_{sid}.txt"

    if out_path.exists():
        log.info("SKIPPED  %s (cache hit)", law_entry["id"])
        return True

    log.info("Fetching  %s — %s", law_entry["id"], law_entry["name"])

    text = crawl(law_entry)

    if text is None:
        try:
            from thuvien_crawler import crawl as thuvien_crawl
            log.info("Trying thuvienphapluat.vn for %s", law_entry["id"])
            text = thuvien_crawl(law_entry)
        except Exception as exc:
            log.warning("thuvien_crawler failed for %s: %s", law_entry["id"], exc)

    if text and len(text) > 500:
        out_path.write_text(text, encoding="utf-8")
        log.info("SUCCESS  %s -> %s (%d chars)", law_entry["id"], out_path.name, len(text))
        return True

    # Both sources failed — write placeholder so TIP-002 can skip gracefully
    todo_path.write_text(f"CRAWL_FAILED: {law_entry['id']}\n", encoding="utf-8")
    log.warning("FAILED   %s — placeholder written", law_entry["id"])
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    laws = config["laws"]
    log.info("Starting crawl for %d documents", len(laws))

    results = {"success": 0, "skipped": 0, "failed": 0}

    for i, entry in enumerate(laws):
        sid = safe_id(entry["id"])
        out_path = OUTPUT_DIR / f"{sid}.txt"

        if out_path.exists():
            results["skipped"] += 1
        elif process_law(entry):
            results["success"] += 1
        else:
            results["failed"] += 1

        if i < len(laws) - 1:
            time.sleep(1)

    log.info(
        "Done — success=%d skipped=%d failed=%d",
        results["success"], results["skipped"], results["failed"],
    )


if __name__ == "__main__":
    main()
