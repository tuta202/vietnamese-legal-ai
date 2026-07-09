from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from legal_rag.backends.gpu import gpu_chat_complete
from legal_rag.retrieval.config import RetrievalConfig


LEGAL_ROUTE = "legal"
SMALL_TALK_ROUTE = "small_talk"
OUT_OF_SCOPE_ROUTE = "out_of_scope"

_VALID_ROUTES = {LEGAL_ROUTE, SMALL_TALK_ROUTE, OUT_OF_SCOPE_ROUTE}

_ROUTER_SYSTEM_PROMPT = """You route user questions for a Vietnamese legal assistant.

Return valid JSON only:
{"route":"legal|small_talk|out_of_scope","reason":"short reason"}

Definitions:
- legal: asks about Vietnamese law, legal rights/duties, procedures, sanctions, taxes, labor, enterprise, bidding, intellectual property, contracts, licenses, government procedures, or asks for legal documents/articles.
- small_talk: greeting, thanks, identity/capability question, or casual conversation not requiring legal retrieval.
- out_of_scope: asks for non-legal advice, coding, math, general knowledge, medicine, finance, travel, entertainment, or anything unrelated to legal help.

If uncertain between legal and another route, choose legal.
Do not answer the question."""

_SMALL_TALK_SYSTEM_PROMPT = """You are a concise Vietnamese legal assistant.
Reply naturally in Vietnamese.
If the user asks what you can do, say you can help answer Vietnamese legal questions with cited legal grounds.
Do not invent legal advice in small talk."""


@dataclass(frozen=True)
class QueryRoute:
    route: str
    reason: str = ""


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def route_query(config: RetrievalConfig, question: str) -> QueryRoute:
    """Route a user query before retrieval.

    The router is intentionally conservative: parse/model failures and unknown
    labels fall back to the legal RAG route, because missing a legal question is
    worse than spending extra retrieval latency.
    """
    try:
        raw = gpu_chat_complete(
            config,
            system=_ROUTER_SYSTEM_PROMPT,
            user=question.strip(),
            temperature=0.0,
            max_tokens=160,
        )
    except Exception:
        return QueryRoute(route=LEGAL_ROUTE, reason="router_error_fallback")
    parsed = _extract_json_object(raw)
    route = str(parsed.get("route", "")).strip().lower()
    if route not in _VALID_ROUTES:
        return QueryRoute(route=LEGAL_ROUTE, reason="router_parse_fallback")
    return QueryRoute(route=route, reason=str(parsed.get("reason", "")).strip())


def answer_small_talk(config: RetrievalConfig, question: str) -> str:
    return gpu_chat_complete(
        config,
        system=_SMALL_TALK_SYSTEM_PROMPT,
        user=question.strip(),
        temperature=0.2,
        max_tokens=320,
    ).strip()


def out_of_scope_answer() -> str:
    return (
        "Mình chỉ hỗ trợ các câu hỏi liên quan đến pháp luật Việt Nam. "
        "Bạn có thể hỏi về quyền, nghĩa vụ, thủ tục, điều kiện, mức xử phạt, "
        "thời hạn hoặc căn cứ pháp lý trong một tình huống cụ thể."
    )
