"""
QueryRewriter — rewrites a Vietnamese legal question into a better retrieval query
and generates a topic description for semantic search.

Shared base class: the model-call hook (`_chat_complete`) is overridden by the
gpu (Gemma) and vertex (Gemini) backends. The base implementation is a generic
OpenAI-compatible call kept only as a fallback.

Strategy: Instead of HyDE (which risks citing outdated law numbers/values),
the second field describes the legal topic, terminology, and relevant document
types — keeping the embedding anchored to concept space, not fabricated facts.

mock=True: skips all network calls; returns the original question for both fields.
The LLM client is initialised lazily on first call to rewrite().
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from legal_rag.retrieval.config import RetrievalConfig


class RewriteResult(TypedDict):
    rewritten_query: str
    topic_description: str


_SYSTEM_PROMPT = """\
Bạn là trợ lý pháp lý Việt Nam. Khi nhận được câu hỏi pháp lý, hãy trả lời ĐÚNG \
theo định dạng JSON sau (không thêm bất kỳ văn bản nào khác):
{
  "rewritten_query": "<câu truy vấn tối ưu cho tìm kiếm ngữ nghĩa, tập trung vào \
khái niệm pháp lý cốt lõi>",
  "topic_description": "<mô tả chủ đề pháp lý: liệt kê chủ đề chính, thuật ngữ pháp lý \
liên quan, loại văn bản có thể chứa câu trả lời (Luật/Nghị định/Thông tư + tên), \
các khái niệm liên quan. KHÔNG đưa số liệu cụ thể (%, số tiền, thời hạn). \
KHÔNG trích dẫn điều khoản cụ thể (Điều X). Chỉ mô tả CHỦ ĐỀ và THUẬT NGỮ.>"
}"""

_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Công ty tôi có 5 người, có phải DNNVV không?",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewritten_query": (
                    "tiêu chí xác định doanh nghiệp nhỏ và vừa "
                    "số lao động vốn doanh thu"
                ),
                "topic_description": (
                    "Chủ đề: tiêu chí phân loại doanh nghiệp nhỏ và vừa. "
                    "Thuật ngữ: doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ, "
                    "doanh nghiệp vừa, số lao động tham gia bảo hiểm xã hội, "
                    "tổng nguồn vốn, tổng doanh thu. "
                    "Văn bản liên quan: Luật Hỗ trợ doanh nghiệp nhỏ và vừa, "
                    "Nghị định hướng dẫn hỗ trợ doanh nghiệp nhỏ và vừa."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": "Người lao động được nghỉ bao nhiêu ngày phép năm?",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewritten_query": (
                    "số ngày nghỉ hàng năm người lao động chế độ nghỉ phép"
                ),
                "topic_description": (
                    "Chủ đề: chế độ nghỉ hằng năm của người lao động. "
                    "Thuật ngữ: nghỉ hằng năm, nghỉ phép năm, lương nguyên, "
                    "điều kiện làm việc nặng nhọc độc hại, thâm niên làm việc. "
                    "Văn bản liên quan: Bộ luật Lao động, Nghị định hướng dẫn "
                    "Bộ luật Lao động về thời giờ làm việc nghỉ ngơi."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": "Thuế GTGT đối với hàng xuất khẩu là bao nhiêu?",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewritten_query": (
                    "thuế suất thuế giá trị gia tăng hàng hóa dịch vụ xuất khẩu"
                ),
                "topic_description": (
                    "Chủ đề: mức thuế suất thuế giá trị gia tăng áp dụng cho "
                    "hàng hóa và dịch vụ xuất khẩu. "
                    "Thuật ngữ: thuế suất ưu đãi xuất khẩu, khu phi thuế quan, "
                    "hàng hóa chịu thuế GTGT, hoàn thuế GTGT hàng xuất khẩu. "
                    "Văn bản liên quan: Luật Thuế giá trị gia tăng, "
                    "Thông tư hướng dẫn thuế GTGT."
                ),
            },
            ensure_ascii=False,
        ),
    },
]

_REPAIR_SYSTEM_PROMPT = """\
Return exactly one valid JSON object for the Vietnamese legal question.
Use two non-empty string fields and no other text:
{
  "rewritten_query": "concise retrieval-ready Vietnamese legal query",
  "topic_description": "Vietnamese legal topic, terminology, and likely document types; no article numbers"
}
"""


class QueryRewriter:
    """
    Rewrites questions for better retrieval using an LLM. Shared base class —
    gpu/vertex subclasses override only `_chat_complete`.
    Initialise with mock=True for offline testing.
    """

    def __init__(self, config: RetrievalConfig, mock: bool = False) -> None:
        self.config  = config
        self.mock    = mock
        self._client = None   # lazy init

    @property
    def _llm(self):
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415 — lazy import
            self._client = OpenAI(
                base_url=self.config.vllm.base_url,
                api_key="dummy",      # vLLM does not check the key
            )
        return self._client

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rewrite(self, question: str) -> RewriteResult:
        """
        Rewrite a question into an optimised query and a topic description.
        Returns {"rewritten_query": ..., "topic_description": ...}.
        """
        if self.mock:
            return RewriteResult(rewritten_query=question, topic_description=question)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(_FEW_SHOTS)
        messages.append({"role": "user", "content": question})

        try:
            raw = self._chat_complete(messages)
            return self._parse_response(raw, question)
        except Exception:
            # Network / parse failure → fall back to original question
            return RewriteResult(rewritten_query=question, topic_description="")

    def rewrite_strict(self, question: str) -> RewriteResult:
        """Rewrite without technical fallbacks; invalid responses raise for resume."""
        if self.mock:
            return RewriteResult(rewritten_query=question, topic_description=question)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(_FEW_SHOTS)
        messages.append({"role": "user", "content": question})
        raw = self._chat_complete(messages)
        try:
            return self._parse_strict_response(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as initial_error:
            repair_messages = [
                {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            repaired_raw = self._chat_complete(repair_messages)
            try:
                return self._parse_strict_response(repaired_raw)
            except (json.JSONDecodeError, TypeError, ValueError) as repair_error:
                raise ValueError(
                    "rewrite response invalid after JSON repair: "
                    f"initial={initial_error}; repair={repair_error}; "
                    f"repair_raw={repaired_raw[:300]!r}"
                ) from repair_error

    @staticmethod
    def _parse_strict_response(raw: str) -> RewriteResult:
        """Parse both required fields without introducing semantic fallbacks."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None
            else:
                data = None
            if data is None:
                rewritten_match = re.search(
                    r'"rewritten_query"\s*:\s*"((?:\\.|[^"\\])*)"',
                    cleaned,
                    flags=re.DOTALL,
                )
                topic_match = re.search(
                    r'"topic_description"\s*:\s*"((?:\\.|[^"\\])*)"',
                    cleaned,
                    flags=re.DOTALL,
                )
                if not rewritten_match or not topic_match:
                    raise ValueError("rewrite response does not contain both required JSON fields")
                data = {
                    "rewritten_query": json.loads(f'"{rewritten_match.group(1)}"'),
                    "topic_description": json.loads(f'"{topic_match.group(1)}"'),
                }
        if not isinstance(data, dict):
            raise ValueError("rewrite response must be a JSON object")
        rewritten_query = str(data.get("rewritten_query") or "").strip()
        topic_description = str(data.get("topic_description") or "").strip()
        if not rewritten_query or not topic_description:
            raise ValueError("rewrite response is missing required fields")
        return RewriteResult(
            rewritten_query=rewritten_query,
            topic_description=topic_description,
        )

    # ------------------------------------------------------------------
    # Model-call hook — overridden by Vertex subclass (only this differs)
    # ------------------------------------------------------------------

    def _chat_complete(self, messages: list[dict]) -> str:
        """Single chat completion over `messages` (system + few-shots + user)."""
        response = self._llm.chat.completions.create(
            model=self.config.vllm.model,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, fallback: str) -> RewriteResult:
        """Robustly parse JSON from LLM output; fall back on any error."""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)

        try:
            data = json.loads(raw.strip())
            return RewriteResult(
                rewritten_query=str(data.get("rewritten_query", fallback)).strip()
                    or fallback,
                topic_description=str(data.get("topic_description", "")).strip(),
            )
        except (json.JSONDecodeError, AttributeError):
            rq = re.search(r'"rewritten_query"\s*:\s*"([^"]+)"', raw)
            td = re.search(r'"topic_description"\s*:\s*"([^"]+)"', raw)
            return RewriteResult(
                rewritten_query=rq.group(1).strip() if rq else fallback,
                topic_description=td.group(1).strip() if td else "",
            )
