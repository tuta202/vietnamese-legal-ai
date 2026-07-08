from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from legal_rag.common.article_lookup import ArticleLookup
from legal_rag.production.single_question_runner import SingleQuestionRagRunner


DEFAULT_CONFIG = "config_gpu_gemini_production.yaml"
DEFAULT_CORPUS = "corpus/data/corpus_clean_asof_20260301.json"
DEFAULT_BM25 = "retrieval/data/bm25_index_asof_20260301.pkl"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_runner() -> SingleQuestionRagRunner:
    return SingleQuestionRagRunner(
        config_path=os.getenv("LEGAL_RAG_CONFIG", DEFAULT_CONFIG),
        corpus_path=os.getenv("LEGAL_RAG_CORPUS", DEFAULT_CORPUS),
        bm25_index_path=os.getenv("LEGAL_RAG_BM25_INDEX", DEFAULT_BM25),
        expected_articles=_env_int("LEGAL_RAG_EXPECTED_ARTICLES", 82570),
        stage1_batch_workers=_env_int("LEGAL_RAG_STAGE1_BATCH_WORKERS", 2),
        final_batch_workers=_env_int("LEGAL_RAG_FINAL_BATCH_WORKERS", 2),
        bge_workers=_env_int("LEGAL_RAG_BGE_WORKERS", 4),
        rescue_coverage_depth=_env_int("LEGAL_RAG_RESCUE_DEPTH", 4),
    )


@lru_cache(maxsize=1)
def get_lookup() -> ArticleLookup:
    return ArticleLookup(os.getenv("LEGAL_RAG_CORPUS", DEFAULT_CORPUS))


def split_answer(answer: str) -> dict[str, str]:
    text = (answer or "").strip()
    if not text:
        return {"conclusion": "", "analysis": "", "has_conclusion": False}

    normalized = re.sub(r"\r\n?", "\n", text)
    conclusion = ""
    analysis = normalized

    conclusion_match = re.search(
        r"(?:^|\n)\s*\*{0,2}Kết luận\*{0,2}\s*\n+(.*?)(?=\n\s*\*{0,2}(?:Căn cứ|Phân tích|Cách áp dụng|Lưu ý)\b|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if conclusion_match:
        conclusion = conclusion_match.group(1).strip(" \n-*")
    else:
        return {"conclusion": "", "analysis": normalized, "has_conclusion": False}

    if conclusion and conclusion in analysis:
        analysis = analysis.replace(conclusion, "", 1).strip()
    return {"conclusion": conclusion, "analysis": analysis, "has_conclusion": bool(conclusion)}


def _doc_ref(article_ref: str) -> str:
    parts = str(article_ref).split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else str(article_ref)


def _anchor_id(index: int) -> str:
    return f"legal-source-{index + 1}"


_DOC_TYPE_PATTERN = (
    r"Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư\s+liên\s+tịch|Thông\s+tư|"
    r"Nghị\s+quyết|Pháp\s+lệnh|Quyết\s+định|Hiến\s+pháp"
)
_ARTICLE_REF_PATTERN = (
    r"(?:điểm\s+[A-Za-zÀ-ỹ]\s+)?(?:khoản\s+\d+[A-Za-zÀ-ỹ]?\s+)?"
    r"Điều\s+\d+[A-Za-zÀ-ỹ]?"
    r"(?:\s*,?\s*khoản\s+\d+[A-Za-zÀ-ỹ]?)?(?:\s*,?\s*điểm\s+[A-Za-zÀ-ỹ])?"
)
_CITATION_RE = re.compile(_ARTICLE_REF_PATTERN, flags=re.IGNORECASE)
_INLINE_STYLE_RE = re.compile(r"(\*\*\*|\*\*|__|`)(.+?)\1")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^\s*(?P<marker>[-*+]|\d+[.)])\s+(?P<text>.+)$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SOURCE_MARKER_RE = re.compile(r"\s*\[(?:A|D|S|R)\d+\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LEGAL_REF_BACK_RE = re.compile(
    rf"(?:{_DOC_TYPE_PATTERN})\s+[^,;.\n]{{0,180}}$",
    flags=re.IGNORECASE,
)
_LEGAL_MARKER_RE = re.compile(
    rf"(?:{_DOC_TYPE_PATTERN})\b",
    flags=re.IGNORECASE,
)
_LEGAL_REF_FORWARD_RE = re.compile(
    rf"^\s*,?\s+(?:của\s+)?(?:{_DOC_TYPE_PATTERN})\s+[^,;.\n]{{0,180}}",
    flags=re.IGNORECASE,
)
_NEXT_LEGAL_REF_RE = re.compile(
    rf"\s+(?:và|hoặc|cùng|theo)\s+(?:(?:{_DOC_TYPE_PATTERN})\b|{_ARTICLE_REF_PATTERN})",
    flags=re.IGNORECASE,
)


def _article_number(article: dict[str, str]) -> str:
    match = re.search(r"\d+[A-Za-z]?", str(article.get("article_number", "")))
    return match.group(0).lower() if match else ""


def _citation_number(text: str) -> str:
    match = re.search(r"Điều\s+(\d+[A-Za-z]?)", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _expand_citation_span(line: str, start: int, end: int) -> tuple[int, int]:
    """Highlight the whole legal reference when the answer writes one compact citation."""
    expanded_start = start
    expanded_end = end

    before = line[max(0, start - 160) : start]
    trimmed_before = before.rstrip(" (")
    marker_matches = list(_LEGAL_MARKER_RE.finditer(trimmed_before))
    if marker_matches:
        last_marker = marker_matches[-1]
        candidate_ref = trimmed_before[last_marker.start() :]
        if _LEGAL_REF_BACK_RE.match(candidate_ref) and not re.search(
            r"\s+(?:và|hoặc|cùng|theo)\s*$",
            candidate_ref,
            flags=re.IGNORECASE,
        ):
            expanded_start = max(0, start - 160) + last_marker.start()

    after = line[end : min(len(line), end + 180)]
    if after.startswith(")") and expanded_start < start and "(" in line[expanded_start:start]:
        expanded_end = end + 1
    elif after.startswith(")") and "(" in line[max(0, start - 80) : start]:
        expanded_end = end + 1
    else:
        forward_match = _LEGAL_REF_FORWARD_RE.match(after)
        if forward_match:
            forward_text = forward_match.group(0)
            next_ref = _NEXT_LEGAL_REF_RE.search(forward_text)
            if next_ref:
                expanded_end = end + next_ref.start()
            else:
                expanded_end = end + forward_match.end()
        elif expanded_start < start:
            close_paren = after.find(")")
            next_punct = min(
                [idx for idx in [after.find(","), after.find(";"), after.find(".")] if idx >= 0],
                default=-1,
            )
            if close_paren >= 0 and (next_punct < 0 or close_paren < next_punct):
                expanded_end = end + close_paren + 1

    return expanded_start, expanded_end


def _best_article_for_citation(text: str, start: int, end: int, articles: list[dict[str, str]]) -> dict[str, str] | None:
    number = _citation_number(text[start:end])
    if not number:
        return None

    candidates = [article for article in articles if _article_number(article) == number]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    window = text[max(0, start - 140) : min(len(text), end + 180)].lower()
    for article in candidates:
        law_id = str(article.get("law_id", "")).lower()
        if law_id and law_id in window:
            return article

    for article in candidates:
        law_name_words = [
            word
            for word in re.split(r"\W+", str(article.get("law_name", "")).lower())
            if len(word) >= 4
        ]
        if law_name_words and sum(1 for word in law_name_words if word in window) >= 3:
            return article
    return candidates[0]


def _append_citation_segments(
    segments: list[dict[str, Any]],
    text: str,
    articles: list[dict[str, str]],
    *,
    kind: str = "text",
) -> None:
    cursor = 0
    for match in _CITATION_RE.finditer(text):
        citation_start, citation_end = _expand_citation_span(text, match.start(), match.end())
        if citation_start < cursor:
            continue
        if citation_start > cursor:
            segments.append({"kind": kind, "text": text[cursor:citation_start]})
        article = _best_article_for_citation(text, match.start(), match.end(), articles)
        if article is None:
            segments.append({"kind": kind, "text": text[citation_start:citation_end]})
        else:
            segments.append(
                {
                    "kind": "citation",
                    "text": text[citation_start:citation_end],
                    "article": article,
                }
            )
        cursor = citation_end
    if cursor < len(text):
        segments.append({"kind": kind, "text": text[cursor:]})


def _append_markdown_segments(
    segments: list[dict[str, Any]],
    text: str,
    articles: list[dict[str, str]],
) -> None:
    cursor = 0
    for match in _INLINE_STYLE_RE.finditer(text):
        if match.start() > cursor:
            _append_citation_segments(segments, text[cursor : match.start()], articles)
        marker = match.group(1)
        kind = "code" if marker == "`" else "bold"
        _append_citation_segments(segments, match.group(2), articles, kind=kind)
        cursor = match.end()
    if cursor < len(text):
        _append_citation_segments(segments, text[cursor:], articles)


def _normalize_markdown_line(line: str) -> str:
    line = _MARKDOWN_LINK_RE.sub(r"\1", line)
    line = _SOURCE_MARKER_RE.sub("", line)
    line = _HTML_TAG_RE.sub("", line)
    return line.strip()


def _strip_surrounding_emphasis(text: str) -> str:
    stripped = text.strip()
    for marker in ("***", "**", "__", "_"):
        if stripped.startswith(marker) and stripped.endswith(marker) and len(stripped) > len(marker) * 2:
            return stripped[len(marker) : -len(marker)].strip()
    return stripped


def _strip_markdown_heading(line: str) -> tuple[int, str]:
    line = _strip_surrounding_emphasis(line)
    match = _HEADING_RE.match(line)
    if not match:
        return 0, line
    return len(match.group(1)), _strip_surrounding_emphasis(match.group(2))


def _strip_markdown_list_marker(line: str) -> tuple[str, str]:
    match = _BULLET_RE.match(line)
    if not match:
        return "", line
    marker = match.group("marker")
    if re.match(r"\d+[.)]", marker):
        return f"{marker} ", match.group("text")
    return "• ", match.group("text")


def build_citation_blocks(text: str, articles: list[dict[str, str]]) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    for line in (text or "").splitlines():
        line = _normalize_markdown_line(line)
        if not line.strip():
            blocks.append([{"kind": "text", "text": ""}])
            continue
        heading_level, line = _strip_markdown_heading(line)
        list_marker, line = _strip_markdown_list_marker(line)
        segments: list[dict[str, Any]] = []
        if heading_level:
            segments.append({"kind": "heading", "text": line, "level": heading_level})
            blocks.append(segments)
            continue
        if list_marker:
            segments.append({"kind": "bullet", "text": list_marker})
        _append_markdown_segments(segments, line, articles)
        blocks.append(segments or [{"kind": "text", "text": line}])
    return blocks


def hydrate_result(result: dict[str, Any]) -> dict[str, Any]:
    lookup = get_lookup()
    article_refs = list(result.get("relevant_articles") or [])
    article_cards: list[dict[str, str]] = []
    docs: dict[str, dict[str, str]] = {}
    missing: list[str] = []

    for article_index, article_ref in enumerate(article_refs):
        article = lookup.get(article_ref)
        if article is None:
            missing.append(article_ref)
            continue
        doc = _doc_ref(article_ref)
        docs.setdefault(
            doc,
            {
                "doc_ref": doc,
                "law_id": str(article.get("law_id", "")),
                "law_type": str(article.get("law_type", "")),
                "law_name": str(article.get("law_name", "")),
            },
        )
        content = " ".join(str(article.get("content", "")).split())
        article_cards.append(
            {
                "article_ref": article_ref,
                "doc_ref": doc,
                "law_id": str(article.get("law_id", "")),
                "law_type": str(article.get("law_type", "")),
                "law_name": str(article.get("law_name", "")),
                "article_number": str(article.get("dieu_number", "")),
                "article_title": str(article.get("dieu_title", "")),
                "anchor_id": _anchor_id(article_index),
                "content": content,
                "content_preview": content[:280] + ("..." if len(content) > 280 else ""),
            }
        )

    answer_parts = split_answer(str(result.get("answer", "")))
    warnings: list[str] = []
    if not article_cards:
        warnings.append("Chưa tìm thấy điều luật đủ tin cậy để grounding câu trả lời.")
    if missing:
        warnings.append(f"Có {len(missing)} điều luật không hydrate được từ corpus local.")
    if len(article_cards) >= 8:
        warnings.append("Câu hỏi có nhiều căn cứ liên quan; nên đọc kỹ phạm vi áp dụng của từng điều.")
    if "thiếu thông tin" in str(result.get("answer", "")).lower():
        warnings.append("Câu trả lời có nêu khả năng thiếu thông tin đầu vào.")

    debug = result.get("_debug") or {}
    sizes = debug.get("sizes") or {}
    return {
        "id": str(result.get("id", "")),
        "question": str(result.get("question", "")),
        "answer": str(result.get("answer", "")),
        "conclusion": answer_parts["conclusion"],
        "analysis": answer_parts["analysis"],
        "has_conclusion": answer_parts["has_conclusion"],
        "conclusion_blocks": build_citation_blocks(answer_parts["conclusion"], article_cards),
        "analysis_blocks": build_citation_blocks(answer_parts["analysis"], article_cards),
        "docs": list(docs.values()),
        "articles": article_cards,
        "warnings": warnings,
        "elapsed_seconds": round(float(debug.get("elapsed_seconds", 0.0)), 2) if debug else 0.0,
        "stage_sizes": sizes,
        "legal_intents": list(debug.get("legal_intents") or []),
    }
