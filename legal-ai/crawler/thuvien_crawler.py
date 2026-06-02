"""
Fallback crawler for https://thuvienphapluat.vn.
Exposes the same crawl(law_entry) -> str | None interface as vbpl_crawler.
"""
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

SEARCH_URL = "https://thuvienphapluat.vn/page/tim-van-ban.aspx"
BASE_URL = "https://thuvienphapluat.vn"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://thuvienphapluat.vn/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_session_initialized = False

# Map law type -> slugs that should appear in the document URL
_TYPE_SLUGS = {
    "Luật": ["luat"],
    "Bộ luật": ["bo-luat", "luat"],
    "Nghị định": ["nghi-dinh"],
    "Thông tư": ["thong-tu"],
}


def _init_session() -> None:
    """Visit the homepage once to acquire session cookies."""
    global _session_initialized
    if _session_initialized:
        return
    try:
        SESSION.get(BASE_URL, timeout=12)
        _session_initialized = True
    except Exception as exc:
        log.debug("tvpl: session init failed: %s", exc)


def _fetch(url: str, params: dict | None = None, retries: int = 3, timeout: int = 20) -> requests.Response | None:
    _init_session()
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            log.warning("tvpl: timeout attempt %d/%d — %s", attempt, retries, url)
            if attempt < retries:
                time.sleep(3)
        except requests.exceptions.HTTPError as exc:
            # 4xx/5xx are definitive — no point retrying
            log.warning("tvpl: HTTP %s for %s — skipping retries", exc.response.status_code, url)
            return None
        except requests.exceptions.RequestException as exc:
            log.warning("tvpl: error attempt %d/%d — %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(3)
    return None


def _full_id_slug(law_id: str) -> str:
    """Convert '59/2020/QH14' -> '59-2020-QH14' for URL matching."""
    clean = re.sub(r"-BLL[ĐD]$", "", law_id)
    return re.sub(r"[/\s]", "-", clean)


def _search_doc_url(law_entry: dict) -> str | None:
    """Search thuvienphapluat.vn and return the URL of the best matching document."""
    law_id = law_entry["id"]
    clean_id = re.sub(r"-BLL[ĐD]$", "", law_id)
    law_type = law_entry.get("type", "")
    law_name = law_entry.get("name", "")

    resp = _fetch(SEARCH_URL, params={"keyword": clean_id})
    if resp is None:
        return None

    soup = BeautifulSoup(resp.content, "lxml")

    # Collect non-tab .aspx links from search results
    candidates = []
    for a in soup.select("a[href*='/van-ban/']"):
        href = a.get("href", "")
        if "?tab=" in href or not href.endswith(".aspx"):
            continue
        if not href.startswith("http"):
            href = BASE_URL + href
        candidates.append((href, a.get_text(strip=True)))

    if not candidates:
        return None

    slug = _full_id_slug(clean_id).lower()  # e.g. "45-2019-qh14"

    # --- Pass 1: full slug + type slug match ---
    type_slugs = _TYPE_SLUGS.get(law_type, [])
    for href, text in candidates:
        href_lower = href.lower()
        if slug in href_lower:
            if not type_slugs or any(ts in href_lower for ts in type_slugs):
                return href

    # --- Pass 2: full slug only (any type) ---
    for href, _ in candidates:
        if slug in href.lower():
            return href

    # --- Pass 3: partial — just the number+year part ---
    parts = clean_id.split("/")
    partial = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else parts[0]
    for href, _ in candidates:
        if partial.lower() in href.lower():
            if not type_slugs or any(ts in href.lower() for ts in type_slugs):
                return href

    # --- Fallback: first candidate ---
    return candidates[0][0]


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract plain text from a thuvienphapluat.vn document page."""
    # Priority order based on observed page structure (divContentDoc = Vietnamese text tab)
    content = (
        soup.find(id="divContentDoc")
        or soup.find(id="tab1")
        or soup.find(id="ctl00_Content_ThongTinVB_divNoiDung")
        or soup.find(id="toanVanWrapper")
        or soup.find(id="divToanVan")
        or soup.select_one("div.content-law")
        or soup.select_one("article")
    )

    if content is None:
        content = soup.body

    if content is None:
        return ""

    for tag in content.find_all(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    lines = []
    for element in content.descendants:
        if element.name in ("p", "div", "h1", "h2", "h3", "h4", "li", "br", "td"):
            text = element.get_text(separator=" ", strip=True)
            if text:
                lines.append(text)

    # Remove consecutive duplicates (layout artifacts)
    cleaned, prev = [], object()
    for line in lines:
        if line != prev:
            cleaned.append(line)
        prev = line

    return "\n".join(cleaned)


def crawl(law_entry: dict) -> str | None:
    """
    Fetch full text for law_entry from thuvienphapluat.vn.
    Uses tvpl_url from config if present, otherwise searches.
    Returns extracted text or None if unavailable.
    """
    law_id = law_entry["id"]

    # Use pinned URL from config if available (avoids search ambiguity)
    if law_entry.get("tvpl_url"):
        doc_url = law_entry["tvpl_url"]
        log.info("tvpl: direct URL for %s", law_id)
    else:
        log.info("tvpl: searching for %s", law_id)
        doc_url = _search_doc_url(law_entry)

    if not doc_url:
        log.warning("tvpl: no result found for %s", law_id)
        return None

    log.info("tvpl: fetching %s", doc_url)
    resp = _fetch(doc_url)
    if resp is None:
        log.warning("tvpl: failed to fetch document page for %s", law_id)
        return None

    soup = BeautifulSoup(resp.content, "lxml")
    text = _extract_text(soup)

    if not text:
        log.warning("tvpl: empty text extracted for %s", law_id)
        return None

    if not re.search(r"Điều\s+\d+", text):
        log.warning("tvpl: no Điều pattern for %s — content may be wrong", law_id)
        return text if len(text) > 1000 else None

    return text
