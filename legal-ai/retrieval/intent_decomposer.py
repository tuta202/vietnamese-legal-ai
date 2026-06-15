"""
Legal Intent Decomposer.

This is intentionally separate from QueryRewriter. The rewrite chain optimizes
global retrieval; this chain produces standalone legal retrieval intents used
only as recall-rescue candidates.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from retrieval.config import RetrievalConfig


@dataclass
class IntentAnalysis:
    intents: list[str]


_MAX_INTENTS = 6

_SYSTEM_PROMPT = """\
You are a Vietnamese legal intent decomposition assistant.

Input: one original Vietnamese legal question.

Output ONLY valid JSON:
{
  "intents": [
    "standalone Vietnamese legal retrieval query"
  ]
}

Rules:
1. Always return at least 1 intent.
2. For simple/single-hop questions, return exactly 1 intent.
3. For multi-hop questions, return 2 to 6 intents.
4. Each intent must be a complete retrieval query usable directly for BM25 and dense retrieval.
5. Do not output keywords only.
6. Do not copy the full original question verbatim.
7. Do not include explanations.
8. Do not include law names unless clearly present or strongly implied by the question.
9. Keep each intent concise but legally meaningful.

An intent is NOT a keyword, entity, topic label, or legal area.
Good intents express one independent legal issue.
"""

_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Pháp luật quy định những phương tiện quảng cáo nào?",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "intents": [
                    "các phương tiện quảng cáo được pháp luật quy định",
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "intents": [
                    "ưu đãi dành cho doanh nghiệp nhỏ và vừa khi tham gia đấu thầu",
                    "cơ chế ưu đãi trong đấu thầu đối với doanh nghiệp nhỏ và vừa",
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            "Đối thủ sao chép trái phép phần mềm để cho thuê thu lợi và làm mất khách hàng; "
            "cần xác định hành vi xâm phạm quyền tác giả ở điểm nào, cách tính tổn thất về "
            "cơ hội kinh doanh ra sao, và phải chuẩn bị tài liệu/chứng cứ gì khi gửi đơn yêu cầu xử lý?"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "intents": [
                    "xác định hành vi xâm phạm quyền tác giả đối với phần mềm",
                    "xác định thiệt hại và tổn thất cơ hội kinh doanh do hành vi xâm phạm quyền tác giả",
                    "hồ sơ tài liệu chứng cứ khi yêu cầu xử lý hành vi xâm phạm quyền tác giả",
                ]
            },
            ensure_ascii=False,
        ),
    },
]


class LegalIntentDecomposer:
    """
    Separate LLM chain that decomposes the original question into retrieval
    intents. It can reuse a backend rewriter's `_chat_complete(messages)` hook,
    but the prompt and call are independent from rewrite().
    """

    def __init__(
        self,
        config: RetrievalConfig,
        mock: bool = False,
        chat_complete: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self.config = config
        self.mock = mock
        self._chat_complete_override = chat_complete
        self._client = None

    @property
    def _llm(self):
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(
                base_url=self.config.vllm.base_url,
                api_key="dummy",
            )
        return self._client

    def decompose(self, question: str) -> IntentAnalysis:
        if self.mock:
            return IntentAnalysis(intents=[question])

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(_FEW_SHOTS)
        messages.append({"role": "user", "content": question})

        try:
            raw = self._chat_complete(messages)
            return self._parse_response(raw, question)
        except Exception:
            return IntentAnalysis(intents=[question])

    def _chat_complete(self, messages: list[dict]) -> str:
        if self._chat_complete_override is not None:
            return self._chat_complete_override(messages)

        response = self._llm.chat.completions.create(
            model=self.config.vllm.model,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str, fallback: str) -> IntentAnalysis:
        data = self._json_from_text(raw)
        intents = self._sanitize_intents(data.get("intents") if isinstance(data, dict) else None)
        return IntentAnalysis(intents=intents or [fallback])

    @staticmethod
    def _json_from_text(raw: str) -> object | None:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _sanitize_intents(raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []

        intents: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = " ".join(str(item or "").split())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            intents.append(text)
            if len(intents) >= _MAX_INTENTS:
                break
        return intents
