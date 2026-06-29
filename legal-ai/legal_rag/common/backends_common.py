"""
backends_common.py — provider-agnostic helpers shared by every model backend
(legal_rag.backends.vertex = Gemini, legal_rag.backends.gpu = GPU endpoints).

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
    "Bạn là chuyên gia pháp lý Việt Nam. Đánh giá mức độ từng điều luật có thể"
    " trả lời trực tiếp câu hỏi.\n\n"
    "Thang điểm 0–10:\n"
    "10 — Điều luật TRỰC TIẾP và ĐẦY ĐỦ trả lời câu hỏi (chứa đúng quy định,"
    " mức phạt, điều kiện hoặc thủ tục được hỏi).\n"
    "7–9 — Điều luật TRỰC TIẾP liên quan và CẦN THIẾT nhưng chỉ trả lời một phần"
    " (cần kết hợp với điều khác mới đầy đủ).\n"
    "4–6 — Liên quan đến CHỦ ĐỀ nhưng KHÔNG trực tiếp trả lời (bối cảnh chung,"
    " không phải căn cứ pháp lý chính).\n"
    "1–3 — Chỉ liên quan GIÁN TIẾP: sai đối tượng áp dụng, sai loại hành vi,"
    " hoặc sai loại văn bản.\n"
    "0 — Không liên quan hoặc sai hoàn toàn.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "- Câu hỏi về đối tượng A, điều luật quy định đối tượng B ≠ A → tối đa 2 điểm.\n"
    "- Câu hỏi về mức phạt, điều luật chỉ định nghĩa hành vi (không có mức phạt)"
    " → tối đa 5 điểm.\n"
    "- Câu hỏi về thủ tục, điều luật chỉ quy định điều kiện → tối đa 5 điểm.\n"
    "- Điều luật tổng quát không được ưu tiên hơn điều luật cụ thể đúng đối tượng.\n\n"
    "CHỈ trả về mảng JSON, không thêm văn bản nào khác:\n"
    '[{"index": 0, "score": 8}, {"index": 1, "score": 3}, ...]'
)
RERANK_SNIPPET = 400   # chars of article content shown to the reranker per item


def build_rerank_blocks(question: str, candidates: list[dict]) -> str:
    """
    Build the rerank user message shown to the LLM judge.

    IDENTICAL across backends (gpu + vertex) so rerank quality is
    backend-independent. Includes the document name (`Văn bản`) per article so the
    model can apply the wrong-entity / wrong-document rules in RERANK_SYSTEM.
    """
    blocks = [f"Câu hỏi: {question}", "", "Danh sách điều luật cần chấm điểm:"]
    for i, c in enumerate(candidates):
        law_name = c.get("law_name", "") or c.get("relevant_doc_str", "")
        blocks.append(
            f"[{i}] Văn bản: {law_name}\n"
            f"Điều: {c.get('dieu_number', '')} — {c.get('dieu_title', '')}\n"
            f"{c.get('content', '')[:RERANK_SNIPPET]}"
        )
    return "\n".join(blocks)


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
    re-minted (GPU endpoints). For backends with non-expiring auth, pass
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
