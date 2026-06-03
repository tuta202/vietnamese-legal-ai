"""
LegalGenerator — generates Vietnamese legal answers via vLLM (OpenAI-compatible API).

mock=True: returns a canned answer that contains Điều X patterns from the input
           articles and the required disclaimer. No network call is made.
vLLM client is initialised lazily on first call to generate().
"""
from __future__ import annotations

import re

from generator.prompt_builder import PromptBuilder, _DISCLAIMER
from retrieval.config import RetrievalConfig

# Pattern: "Điều" + whitespace + digits + optional lowercase letter
# Deliberately case-sensitive so "điều này", "điều kiện" etc. are excluded.
# (Those phrases start with lowercase "điều" in Vietnamese prose, but article
# citations always appear as uppercase "Điều".)
_DIEU_PATTERN = re.compile(r"Điều\s+(\d+[a-zđ]?)", re.UNICODE)


class LegalGenerator:
    """
    Generates legal answers by calling the vLLM chat endpoint.
    Accepts RetrievalConfig; reads vllm.* and generator.* sections.
    """

    def __init__(self, config: RetrievalConfig, mock: bool = False) -> None:
        self.config  = config
        self.mock    = mock
        self._client = None          # lazy init
        self._builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Lazy client property
    # ------------------------------------------------------------------

    @property
    def _llm(self):
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415 — lazy import
            self._client = OpenAI(
                base_url=self.config.vllm.base_url,
                api_key="dummy",   # vLLM does not validate the key
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, question: str, articles: list[dict]) -> str:
        """
        Generate a Vietnamese legal answer for the given question.

        Args:
            question: User's legal question in Vietnamese.
            articles: Retrieved/reranked article dicts from the retrieval stack.

        Returns:
            Answer string in Vietnamese, containing Điều citations + disclaimer.
        """
        if self.mock:
            return self._make_mock_answer(question, articles)

        gen_cfg = self.config.generator
        prompt  = self._builder.build_prompt(
            question, articles, max_articles=gen_cfg.max_articles
        )
        return self._chat_complete(prompt["system"], prompt["user"]).strip()

    # ------------------------------------------------------------------
    # Model-call hook — overridden by Vertex subclass (only this differs)
    # ------------------------------------------------------------------

    def _chat_complete(self, system: str, user: str) -> str:
        """Single chat completion from a system + user prompt pair."""
        gen_cfg = self.config.generator
        response = self._llm.chat.completions.create(
            model=self.config.vllm.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=gen_cfg.temperature,
            max_tokens=gen_cfg.max_tokens,
            top_p=gen_cfg.top_p,
        )
        return response.choices[0].message.content or ""

    def extract_dieu_mentions(self, answer: str) -> list[str]:
        """
        Extract all "Điều X" citations from an answer string.

        Matches "Điều" + whitespace + digits (+ optional letter suffix like 24a).
        Does NOT match "Điều này", "Điều khoản", "Điều kiện", etc. because
        those phrases lack a digit immediately after the whitespace.

        Returns unique mentions in order of first appearance.
        """
        seen:   set[str]  = set()
        result: list[str] = []
        for m in _DIEU_PATTERN.finditer(answer):
            mention = f"Điều {m.group(1)}"
            if mention not in seen:
                seen.add(mention)
                result.append(mention)
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _make_mock_answer(self, question: str, articles: list[dict]) -> str:
        """Return a canned answer containing article citations + disclaimer."""
        if not articles:
            return (
                "Không tìm thấy điều luật liên quan đến câu hỏi của bạn.\n\n"
                + _DISCLAIMER
            )

        lines: list[str] = []

        # Lead with the most relevant article
        first = articles[0]
        dieu     = first.get("dieu_number", "Điều 1")
        law_type = first.get("law_type", "Luật")
        law_name = first.get("law_name", "")
        title    = first.get("dieu_title", "")

        lines.append(
            f"Căn cứ {dieu}, {law_type} {law_name}, {title.lower() or 'quy định như sau'}:"
        )
        lines.append(first.get("content", "")[:300].strip())
        lines.append("")

        # Mention remaining articles (up to 3 more)
        for art in articles[1:4]:
            d  = art.get("dieu_number", "")
            lt = art.get("law_type", "")
            ln = art.get("law_name", "")
            t  = art.get("dieu_title", "")
            if d:
                lines.append(
                    f"Ngoài ra, theo {d}, {lt} {ln}: {t}."
                )

        lines.append("")
        lines.append(_DISCLAIMER)
        return "\n".join(lines)
