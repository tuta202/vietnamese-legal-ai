"""
PromptBuilder — stateless builder for LegalGenerator prompts.

Design goals:
  • Maximise IR recall: instruct the LLM to cite EVERY relevant article
    (F2 score weights recall 4× over precision).
  • Maximise QA quality: enforce accurate citation format, no hallucination,
    practical guidance, and professional-but-accessible Vietnamese writing.
  • No config dependency: stateless, instantiate anywhere.
"""
from __future__ import annotations

_MAX_ARTICLES = 7          # default cap; overridden by GeneratorConfig.max_articles
_CONTENT_TRUNCATE = 600    # chars per article in user prompt

# ---------------------------------------------------------------------------
# System prompt (Vietnamese — enforces citation + completeness + disclaimer)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
Bạn là chuyên gia tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi \
pháp lý dựa HOÀN TOÀN trên các điều luật được cung cấp trong phần ngữ cảnh.

QUY TẮC TRẢ LỜI BẮT BUỘC:

1. CHỈ sử dụng thông tin từ các điều luật được cung cấp. Tuyệt đối không bịa thêm \
điều luật, con số, thời hạn hoặc quy định không có trong ngữ cảnh.

2. LUÔN trích dẫn điều luật theo một trong các định dạng sau:
   • "theo Điều X, [Tên Luật/Nghị định/Thông tư]"
   • "căn cứ Điều X [Tên văn bản]"
   • "theo quy định tại Điều X [Tên văn bản]"

3. ĐỀ CẬP TẤT CẢ các điều luật có liên quan đến câu hỏi — không được bỏ sót bất \
kỳ điều luật nào trong ngữ cảnh. Mỗi điều luật liên quan phải được trích dẫn ít \
nhất một lần trong câu trả lời.

4. Khi nhiều văn bản pháp luật cùng quy định về một vấn đề, hãy nêu rõ mối quan hệ \
giữa chúng (ví dụ: luật gốc và nghị định/thông tư hướng dẫn chi tiết).

5. Trình bày các điều luật theo thứ tự từ quan trọng nhất đến ít quan trọng hơn \
(luật > nghị định > thông tư; điều khoản trực tiếp > điều khoản liên quan).

6. Giải thích các thuật ngữ pháp lý khó hiểu khi cần thiết để người không chuyên \
pháp luật có thể hiểu và áp dụng được.

7. Viết bằng tiếng Việt, giọng văn chuyên nghiệp nhưng dễ tiếp cận, tránh dùng \
văn phong hành chính khô khan quá mức.

8. KẾT THÚC câu trả lời PHẢI có đúng đoạn văn sau (nguyên văn):
   "Lưu ý: Đây là tư vấn sơ bộ dựa trên văn bản pháp luật được cung cấp. Để được \
tư vấn chính xác, vui lòng liên hệ luật sư hoặc cơ quan có thẩm quyền."

9. BẮT BUỘC: Dù câu hỏi hỏi về văn bản, phạm vi áp dụng, hay đối tượng áp dụng, \
câu trả lời PHẢI trích dẫn ít nhất một "Điều X" cụ thể từ văn bản liên quan. \
Ví dụ: "Theo Điều 1 [Tên luật], phạm vi áp dụng bao gồm..."\
"""

_DISCLAIMER = (
    "Lưu ý: Đây là tư vấn sơ bộ dựa trên văn bản pháp luật được cung cấp. "
    "Để được tư vấn chính xác, vui lòng liên hệ luật sư hoặc cơ quan có thẩm quyền."
)


class PromptBuilder:
    """
    Builds system + user prompt dicts for the legal Q&A generator.
    Stateless — safe to share across threads or calls.
    """

    def build_prompt(
        self,
        question: str,
        articles: list[dict],
        max_articles: int = _MAX_ARTICLES,
    ) -> dict[str, str]:
        """
        Build prompt messages for chat completion.

        Args:
            question:     The user's legal question.
            articles:     Retrieved article dicts (each must have law_id, law_type,
                          law_name, dieu_number, dieu_title, content).
            max_articles: Cap on how many articles to include in context.

        Returns:
            {"system": str, "user": str}
        """
        context_articles = articles[:max_articles]
        user_content = self._build_user_content(question, context_articles)
        return {"system": _SYSTEM_PROMPT, "user": user_content}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_user_content(self, question: str, articles: list[dict]) -> str:
        blocks = ["Các điều luật liên quan:"]
        for art in articles:
            blocks.append(self._format_article(art))
        blocks.append(f"\nCâu hỏi: {question}")
        return "\n".join(blocks)

    @staticmethod
    def _format_article(art: dict) -> str:
        law_type = art.get("law_type", "")
        law_id   = art.get("law_id", "")
        dieu_num = art.get("dieu_number", "")
        title    = art.get("dieu_title", "")
        content  = art.get("content", "")[:_CONTENT_TRUNCATE]
        if len(art.get("content", "")) > _CONTENT_TRUNCATE:
            content += "..."

        header = f"[{law_type} {law_id}] {dieu_num}. {title}"
        return f"---\n{header}\n{content}"
