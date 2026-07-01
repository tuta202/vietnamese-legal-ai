from __future__ import annotations

import pytest

from legal_rag.generation import generate_answers
from legal_rag.retrieval.config import RetrievalConfig


def article(index: int, content: str = "Noi dung") -> dict:
    return {
        "law_id": f"{index:02d}/2020/QH14",
        "law_type": "Luat",
        "law_name": f"Van ban {index}",
        "dieu_number": f"\u0110i\u1ec1u {index}",
        "dieu_title": f"Tieu de {index}",
        "content": content,
    }


def test_generation_signature_tracks_article_content_and_config():
    config = RetrievalConfig()
    refs = ["LAW-1|Luat Van ban 1|Dieu 1"]
    first = generate_answers.generation_signature(config, "Cau hoi", refs, [article(1, "A")])
    content_changed = generate_answers.generation_signature(
        config, "Cau hoi", refs, [article(1, "B")]
    )
    config.generator.content_max_chars = 2200
    config_changed = generate_answers.generation_signature(
        config, "Cau hoi", refs, [article(1, "A")]
    )

    assert first != content_changed
    assert first != config_changed


def test_generate_one_passes_every_final_article(monkeypatch):
    captured = {}

    class FakeGenerator:
        def generate(self, question, articles):
            captured["articles"] = articles
            return "Theo \u0110i\u1ec1u 1 Luat thu nghiem (01/2020/QH14)."

    monkeypatch.setattr(generate_answers, "get_worker_generator", lambda _config: FakeGenerator())
    articles = [article(index) for index in range(1, 13)]
    job = {
        "id": 7,
        "question": "Cau hoi",
        "article_refs": [f"ref-{index}" for index in range(12)],
        "articles": articles,
        "signature": "signature",
    }

    result = generate_answers.generate_one(RetrievalConfig(), job)

    assert captured["articles"] == articles
    assert result["article_count"] == 12
    assert "citation_warning" not in result


def test_strict_parser_accepts_supplied_article_pair():
    valid, detail = generate_answers.validate_answer_citations(
        "Theo \u0110i\u1ec1u 1 Luat Van ban 1 (01/2020/QH14).",
        [article(1)],
    )
    assert valid, detail


def test_strict_parser_accepts_law_id_before_article_number():
    valid, detail = generate_answers.validate_answer_citations(
        "Theo Luat Van ban 1 so 01/2020/QH14 (\u0110i\u1ec1u 1).",
        [article(1)],
    )
    assert valid, detail


def test_strict_parser_rejects_wrong_explicit_pair():
    valid, detail = generate_answers.validate_answer_citations(
        "Theo \u0110i\u1ec1u 2 Luat Van ban 1 (01/2020/QH14).",
        [article(1)],
    )
    assert not valid
    assert "not present" in detail


def test_explicit_pair_outside_input_requires_regeneration():
    supplied = article(1)
    requires_regeneration, detail = generate_answers.answer_requires_regeneration(
        "Theo \u0110i\u1ec1u 99 Luat khac (99/2099/QH99).",
        [supplied],
    )
    assert requires_regeneration
    assert "not present" in detail


def test_existing_corpus_pair_outside_input_still_requires_regeneration():
    supplied = article(1)
    requires_regeneration, _detail = generate_answers.answer_requires_regeneration(
        "Theo \u0110i\u1ec1u 2 Luat Van ban 2 (02/2020/QH14).",
        [supplied],
    )
    assert requires_regeneration


def test_guard_depends_on_supplied_input_not_full_corpus_index():
    requires_regeneration, _detail = generate_answers.answer_requires_regeneration(
        "Theo \u0110i\u1ec1u 99 Luat khac (99/2099/QH99).",
        [article(1)],
    )
    assert requires_regeneration


def test_explicit_cross_reference_is_accepted():
    supplied = article(1)
    supplied["content"] = "Tham chieu \u0110i\u1ec1u 99 Luat khac 99/2099/QH99."
    requires_regeneration, _detail = generate_answers.answer_requires_regeneration(
        "Theo \u0110i\u1ec1u 99 Luat khac (99/2099/QH99).",
        [supplied],
    )
    assert not requires_regeneration


def test_conflicting_article_header_metadata_is_accepted():
    supplied = article(1)
    supplied["dieu_title"] = "\u0110i\u1ec1u 2. Tieu de thuc te"
    supplied["content"] = "\u0110i\u1ec1u 2. Noi dung thuc te."
    requires_regeneration, _detail = generate_answers.answer_requires_regeneration(
        "Theo \u0110i\u1ec1u 2 Luat Van ban 1 (01/2020/QH14).",
        [supplied],
    )
    assert not requires_regeneration


def test_unqualified_or_missing_citation_is_not_treated_as_clear_hallucination():
    supplied = article(1)
    for answer in ("Theo \u0110i\u1ec1u 99.", "Theo quy dinh phap luat."):
        requires_regeneration, _detail = generate_answers.answer_requires_regeneration(
            answer,
            [supplied],
        )
        assert not requires_regeneration


def test_empty_answer_requires_regeneration():
    requires_regeneration, detail = generate_answers.answer_requires_regeneration(
        "", [article(1)]
    )
    assert requires_regeneration
    assert detail == "answer is empty"


def test_generate_one_repairs_one_clear_outside_input_citation(monkeypatch):
    captured = {}

    class FakeGenerator:
        def generate(self, question, articles):
            return "Theo \u0110i\u1ec1u 99 Luat khac (99/2099/QH99)."

        def repair_citations(self, question, articles, draft, error, allowed):
            captured["draft"] = draft
            captured["error"] = error
            captured["allowed"] = allowed
            return "Theo \u0110i\u1ec1u 1 Luat Van ban 1 (01/2020/QH14)."

    monkeypatch.setattr(generate_answers, "get_worker_generator", lambda _config: FakeGenerator())
    job = {
        "id": 8,
        "question": "Cau hoi",
        "article_refs": ["ref-1"],
        "articles": [article(1)],
        "known_law_ids": {"01/2020/QH14"},
        "signature": "signature",
    }

    result = generate_answers.generate_one(RetrievalConfig(), job)

    assert "99/2099/QH99" in captured["error"]
    assert captured["allowed"] == ["\u0110i\u1ec1u 1 Van ban 1 (01/2020/QH14)"]
    assert "01/2020/QH14" in result["answer"]


def test_generate_one_accepts_nonempty_answer_after_single_citation_repair(monkeypatch):
    class FakeGenerator:
        def generate(self, question, articles):
            return "Theo \u0110i\u1ec1u 99 Luat khac (99/2099/QH99)."

        def repair_citations(self, question, articles, draft, error, allowed):
            return draft

    monkeypatch.setattr(generate_answers, "get_worker_generator", lambda _config: FakeGenerator())
    job = {
        "id": 9,
        "question": "Cau hoi",
        "article_refs": ["ref-1"],
        "articles": [article(1)],
        "known_law_ids": {"01/2020/QH14"},
        "signature": "signature",
    }

    result = generate_answers.generate_one(RetrievalConfig(), job)

    assert result["citation_repair_applied"] is True
    assert "99/2099/QH99" in result["answer"]
    assert generate_answers.cached_answer_is_acceptable(
        result, job["articles"], job["known_law_ids"]
    )


def test_generate_one_rejects_empty_citation_repair(monkeypatch):
    class FakeGenerator:
        def generate(self, question, articles):
            return "Theo Điều 99 Luat khac (99/2099/QH99)."

        def repair_citations(self, question, articles, draft, error, allowed):
            return ""

    monkeypatch.setattr(generate_answers, "get_worker_generator", lambda _config: FakeGenerator())
    job = {
        "id": 10,
        "question": "Cau hoi",
        "article_refs": ["ref-1"],
        "articles": [article(1)],
        "known_law_ids": {"01/2020/QH14"},
        "signature": "signature",
    }

    with pytest.raises(ValueError, match="empty answer"):
        generate_answers.generate_one(RetrievalConfig(), job)
