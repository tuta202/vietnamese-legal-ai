from legal_rag.verification import candidate_verifier as llm_candidate_verifier
from legal_rag.verification import final_collective
import pytest
from legal_rag.verification.final_collective import (
    alias_candidates as alias_collective_candidates,
    process_question as process_collective_question,
)
from legal_rag.verification.candidate_verifier import (
    alias_candidates,
    parse_stage1_alias_json,
    verify_one,
)
from legal_rag.retrieval.embedder import embedding_text_sha256, format_document_text


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


@pytest.mark.parametrize(
    "raw",
    [
        '{"confidence":"high"}',
        '{"selected_article_keys":"A1","confidence":"high"}',
        '{"selected_article_keys":["A99"],"confidence":"high"}',
    ],
)
def test_alias_parser_rejects_schema_and_unknown_key_errors(raw):
    parsed, ok = parse_stage1_alias_json(raw, {"A1": "article-1"})
    assert parsed is None
    assert ok is False


def test_stage1_repairs_malformed_json_without_fallback(monkeypatch):
    calls = []

    class Worker:
        def call(self, question, candidate_articles, *, system_prompt, legal_intents):
            calls.append(system_prompt)
            if len(calls) == 1:
                return '{"selected": ["A1"], "confidence": "high"}'
            return '{"selected_article_keys":["A1"],"confidence":"high"}'

    class Lookup:
        def verifier_candidate(self, article_id, content_max_chars):
            item = candidate()
            item["article_id"] = article_id
            return item

    monkeypatch.setattr(llm_candidate_verifier, "get_worker", lambda _config: Worker())

    result = verify_one(
        object(),
        Lookup(),
        {"id": 1, "question": "Cau hoi", "relevant_articles": ["article-1"]},
        ["intent"],
        strict_errors=True,
        stage1_compact_candidates=True,
    )

    assert result["final_article_ids"] == ["article-1"]
    assert result["batch_results"][0]["repair_attempted"] is True
    assert result["batch_results"][0]["parse_ok"] is True
    assert result["fallback_used"] is True  # Existing one-selection evidence guard.


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


def test_final_collective_trusts_single_selection_without_top1_rescue(monkeypatch):
    class Worker:
        def call(self, question, candidate_articles, *, system_prompt, legal_intents):
            assert all(set(item) == {"key", "source", "article", "content"} for item in candidate_articles)
            return '{"selected_article_keys":["A2"],"confidence":"high"}'

    class Lookup:
        def verifier_candidate(self, article_id, content_max_chars):
            item = candidate()
            item["article_id"] = article_id
            return item

    monkeypatch.setattr(final_collective, "get_worker", lambda _config: Worker())

    result = process_collective_question(
        object(),
        Lookup(),
        {
            "id": 1,
            "question": "Cau hoi",
            "final_article_ids": ["article-1", "article-2", "article-3"],
        },
        ["intent"],
        content_max_chars=2200,
        batch_size=6,
        direct_max=8,
        min_size=3,
        preserve_top1=False,
        strict_errors=True,
        system_prompt=final_collective.SYSTEM_PROMPT,
        prompt_mode="final_precision_gemma_v2",
        compact_candidates=True,
    )

    assert result["final_article_ids"] == ["article-2"]
    assert result["collective_filter"]["fallback_used"] is False


def test_final_collective_uses_shortlist_prompt_only_for_batch_rounds(monkeypatch):
    prompts = []

    class Worker:
        def call(self, question, candidate_articles, *, system_prompt, legal_intents):
            prompts.append(system_prompt)
            return '{"selected_article_keys":["A1"],"confidence":"high"}'

    class Lookup:
        def verifier_candidate(self, article_id, content_max_chars):
            item = candidate()
            item["article_id"] = article_id
            return item

    monkeypatch.setattr(final_collective, "get_worker", lambda _config: Worker())
    process_collective_question(
        object(),
        Lookup(),
        {"id": 1, "question": "Cau hoi", "final_article_ids": [f"article-{i}" for i in range(9)]},
        ["intent"],
        content_max_chars=2200,
        batch_size=6,
        direct_max=8,
        min_size=2,
        preserve_top1=False,
        strict_errors=True,
        system_prompt=final_collective.SYSTEM_PROMPT,
        prompt_mode="final_precision_gemma_v4",
        compact_candidates=True,
    )

    assert prompts[:-1] == [final_collective.BATCH_SYSTEM_PROMPT] * 2
    assert prompts[-1] == final_collective.SYSTEM_PROMPT


def test_strict_stage1_rejects_missing_hydration():
    class MissingLookup:
        def verifier_candidate(self, article_id, content_max_chars):
            return None

    with pytest.raises(RuntimeError, match="stage1 missing 1 hydrated articles"):
        verify_one(
            object(),
            MissingLookup(),
            {"id": 1, "question": "Cau hoi", "relevant_articles": ["article-1"]},
            ["intent"],
            strict_errors=True,
        )


def test_strict_collective_rejects_missing_hydration():
    class MissingLookup:
        def verifier_candidate(self, article_id, content_max_chars):
            return None

    with pytest.raises(RuntimeError, match="collective missing 1 hydrated articles"):
        process_collective_question(
            object(),
            MissingLookup(),
            {"id": 1, "question": "Cau hoi", "final_article_ids": ["article-1"]},
            ["intent"],
            content_max_chars=2200,
            batch_size=6,
            direct_max=8,
            min_size=1,
            preserve_top1=False,
            strict_errors=True,
            system_prompt="prompt",
            prompt_mode="final_precision",
            compact_candidates=False,
        )


def test_embedding_fingerprint_tracks_exact_formatted_text():
    article = {
        "law_type": "Luat",
        "law_name": "Ten luat",
        "dieu_number": "Dieu 1",
        "dieu_title": "Tieu de",
        "content": "Noi dung",
    }
    assert format_document_text(article) == "Luat Ten luat\nDieu 1. Tieu de\nNoi dung"
    original_hash = embedding_text_sha256(article)
    article["content"] = "Noi dung da thay doi"
    assert embedding_text_sha256(article) != original_hash
