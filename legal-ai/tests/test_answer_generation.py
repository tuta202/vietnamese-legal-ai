from __future__ import annotations

from legal_rag.generation import generate_answers
from legal_rag.retrieval.config import RetrievalConfig


def article(index: int, content: str = "Nội dung") -> dict:
    return {
        "law_id": f"LAW-{index}",
        "law_type": "Luật",
        "law_name": f"Văn bản {index}",
        "dieu_number": f"Điều {index}",
        "dieu_title": f"Tiêu đề {index}",
        "content": content,
    }


def test_generation_signature_tracks_article_content_and_config():
    config = RetrievalConfig()
    refs = ["LAW-1|Luật Văn bản 1|Điều 1"]
    first = generate_answers.generation_signature(config, "Câu hỏi", refs, [article(1, "A")])
    content_changed = generate_answers.generation_signature(
        config, "Câu hỏi", refs, [article(1, "B")]
    )
    config.generator.content_max_chars = 2200
    config_changed = generate_answers.generation_signature(
        config, "Câu hỏi", refs, [article(1, "A")]
    )

    assert first != content_changed
    assert first != config_changed


def test_answer_requires_explicit_article_citation():
    assert generate_answers.answer_is_valid("Theo Điều 5 Luật Doanh nghiệp...")
    assert not generate_answers.answer_is_valid("Theo pháp luật hiện hành...")
    assert not generate_answers.answer_is_valid("")


def test_generate_one_passes_every_final_article(monkeypatch):
    captured = {}

    class FakeGenerator:
        def generate(self, question, articles):
            captured["question"] = question
            captured["articles"] = articles
            return "Theo Điều 1 Luật thử nghiệm, quy định được áp dụng."

    monkeypatch.setattr(generate_answers, "get_worker_generator", lambda _config: FakeGenerator())
    articles = [article(index) for index in range(1, 13)]
    job = {
        "id": 7,
        "question": "Câu hỏi nhiều điều luật",
        "article_refs": [f"ref-{index}" for index in range(12)],
        "articles": articles,
        "signature": "signature",
    }

    result = generate_answers.generate_one(RetrievalConfig(), job)

    assert captured["articles"] == articles
    assert len(captured["articles"]) == 12
    assert result["article_count"] == 12
    assert result["signature"] == "signature"
