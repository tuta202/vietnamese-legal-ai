"""
QueryRewriter — rewrites a Vietnamese legal question into a better retrieval query
and generates a topic description for semantic search, using an LLM via vLLM.

Strategy: Instead of HyDE (which risks citing outdated law numbers/values),
the second field describes the legal topic, terminology, and relevant document
types — keeping the embedding anchored to concept space, not fabricated facts.

mock=True: skips all network calls; returns the original question for both fields.
vLLM client is initialised lazily on first call to rewrite().
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from retrieval.config import RetrievalConfig


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


class QueryRewriter:
    """
    Rewrites questions for better retrieval using an LLM (vLLM backend).
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
            response = self._llm.chat.completions.create(
                model=self.config.vllm.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_response(raw, question)
        except Exception:
            # Network / parse failure → fall back to original question
            return RewriteResult(rewritten_query=question, topic_description="")

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
