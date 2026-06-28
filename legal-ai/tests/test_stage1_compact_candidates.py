import llm_candidate_verifier
import pytest
from llm_best_collective_filter import alias_candidates as alias_collective_candidates
from llm_candidate_verifier import (
    alias_candidates,
    parse_stage1_alias_json,
    run_stage2_call,
    stage2_minimal_select,
)


def candidate():
    return {
        "article_id": "123/2020/ND-CP|Nghi dinh Quy dinh ve hoa don|Dieu 15",
        "law_type": "Nghi dinh",
        "law_name": "Quy dinh ve hoa don, chung tu",
        "article_number": "Dieu 15",
        "article_title": "Dang ky su dung hoa don dien tu",
        "article_content": "Noi dung dieu luat",
    }


def test_compact_candidate_shape():
    aliased, key_to_id = alias_candidates([candidate()], compact=True)

    assert key_to_id == {
        "A1": "123/2020/ND-CP|Nghi dinh Quy dinh ve hoa don|Dieu 15"
    }
    assert aliased == [
        {
            "key": "A1",
            "source": "123/2020/ND-CP - Quy dinh ve hoa don, chung tu",
            "article": "Dieu 15. Dang ky su dung hoa don dien tu",
            "content": "Noi dung dieu luat",
        }
    ]


def test_default_candidate_shape_is_unchanged():
    aliased, _ = alias_candidates([candidate()])

    assert aliased[0]["article_key"] == "A1"
    assert aliased[0]["law_type"] == "Nghi dinh"
    assert aliased[0]["article_content"] == "Noi dung dieu luat"
    assert "key" not in aliased[0]


def test_old_parser_maps_compact_key():
    parsed, ok = parse_stage1_alias_json(
        '{"selected_article_keys":["A1"],"confidence":"high"}',
        {"A1": "article-1"},
    )

    assert ok is True
    assert parsed["selected_article_ids"] == ["article-1"]
    assert parsed["confidence"] == "high"


def test_stage2_compact_payload_maps_alias_back_to_article_id(monkeypatch):
    captured = {}

    class Worker:
        def call(self, question, candidate_articles, *, system_prompt, legal_intents):
            captured["candidate_articles"] = candidate_articles
            return '{"selected_article_keys":["A1"],"confidence":"high"}'

    monkeypatch.setattr(llm_candidate_verifier, "get_worker", lambda _config: Worker())

    result = run_stage2_call(
        object(),
        "Cau hoi",
        [candidate()],
        ["Y dinh phap ly"],
        compact_candidates=True,
    )

    assert captured["candidate_articles"] == [
        {
            "key": "A1",
            "source": "123/2020/ND-CP - Quy dinh ve hoa don, chung tu",
            "article": "Dieu 15. Dang ky su dung hoa don dien tu",
            "content": "Noi dung dieu luat",
        }
    ]
    assert result["parse_ok"] is True
    assert result["selected_article_keys"] == ["A1"]
    assert result["selected_article_ids"] == [candidate()["article_id"]]


def test_stage2_strict_errors_does_not_fallback_or_cache(monkeypatch):
    class FailingWorker:
        def call(self, *args, **kwargs):
            raise RuntimeError("temporary API error")

    class Lookup:
        def verifier_candidate(self, article_id, content_max_chars):
            item = candidate()
            item["article_id"] = article_id
            return item

    monkeypatch.setattr(llm_candidate_verifier, "get_worker", lambda _config: FailingWorker())

    with pytest.raises(RuntimeError, match="stage2 technical issue"):
        stage2_minimal_select(
            object(),
            Lookup(),
            question="Cau hoi",
            stage1_article_ids=["article-1", "article-2"],
            legal_intents=["Y dinh phap ly"],
            compact_candidates=True,
            strict_errors=True,
        )


def test_final_collective_compact_candidate_shape():
    aliased, key_to_id = alias_collective_candidates([candidate()], compact=True)

    assert key_to_id == {"A1": candidate()["article_id"]}
    assert aliased == [
        {
            "key": "A1",
            "source": "123/2020/ND-CP - Quy dinh ve hoa don, chung tu",
            "article": "Dieu 15. Dang ky su dung hoa don dien tu",
            "content": "Noi dung dieu luat",
        }
    ]
