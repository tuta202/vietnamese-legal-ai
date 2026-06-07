"""
backends_common.py — provider-agnostic helpers shared by every model backend
(vertex_backends.py = Gemini, garden_backends.py = Model Garden).

Keeping these here means each backend module only contains its own model-call
code; the retry policy, JSON-array parsing and the reranking prompt are defined
once and composed in. Pure-Python — importing this never pulls in any heavy SDK.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

# ---------------------------------------------------------------------------
# Reranking prompt (LLM judges 0–10 relevance, returns a JSON array)
# ---------------------------------------------------------------------------

RERANK_SYSTEM = (
    "Bạn là chuyên gia pháp lý Việt Nam. Chấm điểm mức độ liên quan của từng "
    "điều luật đối với câu hỏi, thang điểm 0 đến 10 (10 = liên quan trực tiếp "
    "nhất). CHỈ trả về một mảng JSON, không thêm văn bản nào khác, đúng định dạng:\n"
    '[{"index": 0, "score": 8}, {"index": 1, "score": 3}]'
)
RERANK_SNIPPET = 400   # chars of article content shown to the reranker per item


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def strip_code_fences(raw: str) -> str:
    """Remove ```json … ``` fences (and stray prose markers) around a payload."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def parse_rerank_scores(raw: str) -> dict[int, float] | None:
    """
    Parse an LLM rerank response into {candidate_index: score}.
    Tolerates code fences and surrounding prose. Returns None on any failure so
    callers can fall back to retrieval order.
    """
    cleaned = strip_code_fences(raw)
    m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(0)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    out: dict[int, float] = {}
    for item in data:
        try:
            out[int(item["index"])] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return out or None


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

def retry_transient(
    fn: Callable,
    *,
    attempts: int = 5,
    base: float = 4.0,
    max_sleep: float = 60.0,
    refresh_auth: Callable[[], None] | None = None,
):
    """
    Call fn(); on transient errors retry with exponential backoff (capped
    max_sleep). Non-transient errors propagate immediately.

    Transient = rate limits (429 / RESOURCE_EXHAUSTED), server errors
    (500/503 / UNAVAILABLE), and connection-level failures (dropped/reset
    connections, read timeouts) — these are common on long runs against a remote
    endpoint and must be ridden out, not fatal. Auth expiry (401 /
    UNAUTHENTICATED) is treated as transient ONLY when `refresh_auth` is supplied
    — it is invoked before the next attempt so a stale bearer token can be
    re-minted (Model Garden). For backends with non-expiring auth, pass
    refresh_auth=None and 401 propagates.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            is_auth = "401" in msg or code == 401 or "UNAUTHENTICATED" in msg
            # Connection-level errors carry no HTTP code — match by exception
            # class name + message so we stay dependency-free (no requests import).
            name = type(e).__name__
            is_conn = (
                name in ("ConnectionError", "Timeout", "ReadTimeout",
                         "ConnectTimeout", "ConnectionResetError",
                         "ProtocolError", "RemoteDisconnected",
                         "ChunkedEncodingError")
                or "Connection aborted" in msg or "RemoteDisconnected" in msg
                or "Connection reset" in msg or "timed out" in msg
                or "Max retries exceeded" in msg
                or "Connection refused" in msg
            )
            transient = (
                "429" in msg or "RESOURCE_EXHAUSTED" in msg
                or "503" in msg or "500" in msg or "UNAVAILABLE" in msg
                or (isinstance(code, int) and code in (429, 500, 503))
                or (is_auth and refresh_auth is not None)
                or is_conn
            )
            if not transient:
                raise
            if is_auth and refresh_auth is not None:
                refresh_auth()
            last = e
            time.sleep(min(base * (2 ** i), max_sleep))
    if last:
        raise last
