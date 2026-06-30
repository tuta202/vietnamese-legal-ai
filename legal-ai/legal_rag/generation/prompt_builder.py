"""Build compact, grounded prompts for Vietnamese legal answer generation."""
from __future__ import annotations

_MAX_ARTICLES = 0  # 0 means include every selected article.
_CONTENT_MAX_CHARS = 2400
_TOTAL_CONTENT_MAX_CHARS = 48000
PROMPT_VERSION = "gemma3-grounded-answer-v3-structure-guard-t0"

_SYSTEM_PROMPT = """\
You answer Vietnamese legal questions from the supplied legal articles.

Rules:
1. Use only the supplied articles. Do not invent legal rules, facts, numbers, deadlines, authorities, or citations.
2. Answer every part of the question. For multi-issue questions, cover each issue separately.
3. State the practical conclusion first, then explain conditions, exceptions, procedure, documents, deadlines, authority, consequences, or penalties when supported and relevant.
4. Cite the exact legal basis for each material conclusion using: "Theo Điều X [tên văn bản] (số/ký hiệu văn bản)". Never cite an article for a rule that it does not contain.
5. Use every supplied article that materially supports the answer, but do not mention an article merely because it is present in the context.
6. If a general law and a decree/circular provide different useful details, explain their roles. If provisions appear to conflict or the evidence is insufficient, say so instead of guessing.
7. Treat every numbered khoản/điểm/item as a separate rule. Attach a condition, exception, penalty, remedy, interest payment, deadline, or consequence to an act only when the same khoản/điểm explicitly links them, or an explicit cross-reference links them.
8. Never transfer a consequence from a neighboring khoản/điểm to the act being asked about. If the supplied text does not explicitly connect them, omit that consequence or state that the evidence does not establish it.
9. The fixed article number is shown in each "Article (fixed citation)" header. Numbers inside Content are khoản/điểm/items, not article numbers. Never turn "khoản 2" into "Điều 2". Cite only the header article number, then add khoản/điểm when supported.
10. Keep penalty amounts and remedies separate. Do not claim that a general remedy article sets a fine amount, and do not apply an interest remedy to documents when it is written only for money or property.
11. Write in clear, concise Vietnamese for a non-lawyer. Prefer short sections or numbered steps when they improve readability.

Suggested answer structure (adapt it to the question):
- Kết luận
- Căn cứ và phân tích
- Cách áp dụng thực tế

Do not discuss the retrieval process or the supplied context.
"""

_DISCLAIMER = (
    "Lưu ý: Nội dung trên được tổng hợp từ các văn bản pháp luật được cung cấp; "
    "trường hợp cụ thể nên được đối chiếu thêm với cơ quan có thẩm quyền hoặc chuyên gia pháp lý."
)


class PromptBuilder:
    """Stateless prompt builder, safe to share across worker threads."""

    def build_prompt(
        self,
        question: str,
        articles: list[dict],
        max_articles: int = _MAX_ARTICLES,
        content_max_chars: int = _CONTENT_MAX_CHARS,
        total_content_max_chars: int = _TOTAL_CONTENT_MAX_CHARS,
    ) -> dict[str, str]:
        """Build a grounded generation prompt.

        ``max_articles <= 0`` includes every selected article. The total content
        budget is shared fairly so a large candidate set cannot silently remove
        later articles from the prompt.
        """
        context_articles = articles if max_articles <= 0 else articles[:max_articles]
        article_count = len(context_articles)
        if article_count:
            fair_share = max(1, total_content_max_chars // article_count)
            per_article_chars = min(content_max_chars, fair_share)
        else:
            per_article_chars = content_max_chars

        user_content = self._build_user_content(
            question,
            context_articles,
            content_max_chars=per_article_chars,
        )
        return {"system": _SYSTEM_PROMPT, "user": user_content}

    def _build_user_content(
        self,
        question: str,
        articles: list[dict],
        *,
        content_max_chars: int,
    ) -> str:
        blocks = [f"QUESTION:\n{question}", "LEGAL ARTICLES:"]
        for index, article in enumerate(articles, start=1):
            blocks.append(self._format_article(article, index, content_max_chars))
        blocks.append("Write the final answer in Vietnamese.")
        return "\n\n".join(blocks)

    @staticmethod
    def _content_excerpt(content: str, max_chars: int) -> str:
        content = " ".join(str(content or "").split())
        if len(content) <= max_chars:
            return content
        marker = "\n[... phần giữa được rút gọn ...]\n"
        if max_chars <= len(marker) + 40:
            return content[:max_chars]

        # Preserve the operative opening and late exceptions/conclusions.
        available_chars = max_chars - len(marker)
        tail_chars = max(20, available_chars // 4)
        head_chars = available_chars - tail_chars
        return (
            content[:head_chars].rstrip()
            + marker
            + content[-tail_chars:].lstrip()
        )

    @classmethod
    def _format_article(cls, article: dict, index: int, content_max_chars: int) -> str:
        law_type = str(article.get("law_type", "")).strip()
        law_name = str(article.get("law_name", "")).strip()
        law_id = str(article.get("law_id", "")).strip()
        article_number = str(article.get("dieu_number", "")).strip()
        article_title = str(article.get("dieu_title", "")).strip()
        content = cls._content_excerpt(article.get("content", ""), content_max_chars)

        source = " ".join(part for part in (law_type, law_name) if part)
        if law_id:
            source = f"{source} ({law_id})" if source else law_id
        article_heading = ". ".join(part for part in (article_number, article_title) if part)
        return (
            f"[A{index}]\nSource: {source}\n"
            f"Article (fixed citation): {article_heading}\nContent: {content}"
        )
