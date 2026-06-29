"""
End-to-end tests for the pipeline and submission modules.
All tests run OFFLINE (mock=True); no GPU/Docker/vLLM needed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# conftest.py ensures legal-ai/ is in sys.path
from legal_rag.pipeline import LegalAIPipeline, PipelineState
from legal_rag.output.submission import save_submission, validate_submission

_TESTS_DIR = Path(__file__).resolve().parent
_SAMPLE_QS = _TESTS_DIR / "sample_questions.json"


# ---------------------------------------------------------------------------
# Shared fixture: mock pipeline (built once per session for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_pipeline() -> LegalAIPipeline:
    """Mock pipeline — BM25 real, all neural models mocked."""
    return LegalAIPipeline(mock=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_entry(q_id: int = 1) -> dict:
    return {
        "id":                q_id,
        "question":          "Câu hỏi mẫu?",
        "answer":            "Căn cứ Điều 4, Luật DNNVV, quy định như sau. "
                             "Lưu ý: Đây là tư vấn sơ bộ.",
        "relevant_docs":     ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
        "relevant_articles": ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"],
    }


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipelineMock:
    def test_single_question_output_schema(self, mock_pipeline):
        entry = mock_pipeline.process_question(1, "Doanh nghiệp nhỏ là gì?")
        assert set(entry.keys()) == {
            "id", "question", "answer", "relevant_docs", "relevant_articles"
        }

    def test_single_question_id_preserved(self, mock_pipeline):
        entry = mock_pipeline.process_question(42, "Thuế GTGT là gì?")
        assert entry["id"] == 42

    def test_single_question_answer_has_dieu(self, mock_pipeline):
        entry = mock_pipeline.process_question(1, "Quyền của doanh nghiệp?")
        assert re.search(r"Điều\s+\d+", entry["answer"]), (
            f"No Điều X in answer: {entry['answer'][:200]}"
        )

    def test_single_question_relevant_docs_deduped(self, mock_pipeline):
        entry = mock_pipeline.process_question(1, "Đăng ký doanh nghiệp?")
        assert len(entry["relevant_docs"]) == len(set(entry["relevant_docs"]))

    def test_single_question_relevant_articles_deduped(self, mock_pipeline):
        entry = mock_pipeline.process_question(1, "Đăng ký doanh nghiệp?")
        assert len(entry["relevant_articles"]) == len(set(entry["relevant_articles"]))

    def test_relevant_articles_format(self, mock_pipeline):
        entry = mock_pipeline.process_question(1, "Thuế TNDN?")
        for art in entry["relevant_articles"]:
            assert "|" in art, f"Missing | separator: {art}"
            parts = art.split("|")
            assert len(parts) == 3, f"Expected 3 pipe-parts: {art}"
            assert re.search(r"Điều\s+\d+", parts[2]), f"Third part not Điều X: {art}"

    def test_mock_batch_three_questions(self, mock_pipeline):
        questions = [
            {"id": 1, "question": "Doanh nghiệp nhỏ và vừa?"},
            {"id": 2, "question": "Thuế GTGT?"},
            {"id": 3, "question": "Đăng ký doanh nghiệp?"},
        ]
        results = mock_pipeline.process_questions(questions)
        assert len(results) == 3
        assert [r["id"] for r in results] == [1, 2, 3]
        for r in results:
            assert "answer" in r

    def test_process_batch_from_file(self, mock_pipeline):
        if not _SAMPLE_QS.exists():
            pytest.skip("sample_questions.json not found")
        results = mock_pipeline.process_batch(_SAMPLE_QS)
        assert len(results) == 3
        for r in results:
            assert set(r.keys()) >= {"id", "question", "answer",
                                     "relevant_docs", "relevant_articles"}

    def test_step_methods_exist(self, mock_pipeline):
        for method in ["step_rewrite", "step_retrieve", "step_rerank",
                       "step_generate", "step_format"]:
            assert hasattr(mock_pipeline, method), f"Missing step method: {method}"
            assert callable(getattr(mock_pipeline, method))

    def test_step_rewrite_sets_state_fields(self, mock_pipeline):
        state = PipelineState(question_id=1, question="Câu hỏi?")
        state = mock_pipeline.step_rewrite(state)
        assert state.rewritten_query != ""
        assert state.topic_description != ""

    def test_no_model_loaded_in_mock_mode(self, mock_pipeline):
        assert mock_pipeline._embedder._model is None
        assert mock_pipeline._rewriter._client is None
        assert mock_pipeline._reranker._model is None
        assert mock_pipeline._generator._client is None


# ---------------------------------------------------------------------------
# submit.py — validate_submission
# ---------------------------------------------------------------------------

class TestValidateSubmission:
    def test_valid_submission_passes(self):
        results = [_make_valid_entry(i) for i in range(1, 4)]
        valid, errors = validate_submission(results)
        assert valid is True
        assert errors == []

    def test_missing_field_flagged(self):
        bad = {"id": 1, "question": "q", "answer": "Điều 1", "relevant_docs": []}
        # missing "relevant_articles"
        _, errors = validate_submission([bad])
        assert any("relevant_articles" in e for e in errors)

    def test_answer_missing_dieu_flagged(self):
        entry = _make_valid_entry()
        entry["answer"] = "Không có trích dẫn điều luật nào."
        _, errors = validate_submission([entry])
        assert any("Điều X" in e for e in errors)

    def test_bad_relevant_article_format_flagged(self):
        entry = _make_valid_entry()
        entry["relevant_articles"] = ["just a string without pipes"]
        _, errors = validate_submission([entry])
        assert any("relevant_article" in e for e in errors)

    def test_bad_relevant_doc_format_flagged(self):
        entry = _make_valid_entry()
        entry["relevant_docs"] = ["no-pipe-here"]
        _, errors = validate_submission([entry])
        assert any("relevant_doc" in e for e in errors)

    def test_duplicate_ids_flagged(self):
        results = [_make_valid_entry(1), _make_valid_entry(1)]
        _, errors = validate_submission(results)
        assert any("duplicate" in e.lower() for e in errors)

    def test_non_integer_id_flagged(self):
        entry = _make_valid_entry()
        entry["id"] = "abc"
        entry["answer"] = "Điều 1 áp dụng."
        _, errors = validate_submission([entry])
        assert any("integer" in e.lower() for e in errors)

    def test_empty_list_is_valid(self):
        valid, errors = validate_submission([])
        assert valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# submit.py — save_submission
# ---------------------------------------------------------------------------

class TestSaveSubmission:
    def test_zip_created_in_tmpdir(self, tmp_path):
        results = [_make_valid_entry(i) for i in range(1, 4)]
        zip_path = save_submission(results, tmp_path)
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

    def test_zip_contains_only_results_json(self, tmp_path):
        import zipfile as zf
        results = [_make_valid_entry(1)]
        zip_path = save_submission(results, tmp_path)
        with zf.ZipFile(zip_path) as z:
            assert z.namelist() == ["results.json"]

    def test_json_inside_zip_is_valid(self, tmp_path):
        import zipfile as zf
        results = [_make_valid_entry(i) for i in range(1, 4)]
        zip_path = save_submission(results, tmp_path)
        with zf.ZipFile(zip_path) as z:
            data = json.loads(z.read("results.json").decode("utf-8"))
        assert len(data) == 3
        assert all("id" in r for r in data)

    def test_results_json_also_written(self, tmp_path):
        results = [_make_valid_entry(1)]
        save_submission(results, tmp_path)
        json_path = tmp_path / "results.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(data) == 1


# ---------------------------------------------------------------------------
# AC-2: full mock pipeline → results.json
# ---------------------------------------------------------------------------

class TestEndToEndMock:
    def test_full_pipeline_produces_valid_submission(self, mock_pipeline, tmp_path):
        """AC-2/3/4 combined: pipeline → results.json → validate → ZIP."""
        questions = [
            {"id": 1, "question": "Điều kiện để được hỗ trợ doanh nghiệp nhỏ?"},
            {"id": 2, "question": "Thuế GTGT hàng xuất khẩu?"},
            {"id": 3, "question": "Cách đăng ký thành lập công ty?"},
        ]
        results = mock_pipeline.process_questions(questions)

        # results.json schema
        assert len(results) == 3
        for r in results:
            assert set(r.keys()) >= {
                "id", "question", "answer", "relevant_docs", "relevant_articles"
            }

        # All answers have Điều X
        for r in results:
            assert re.search(r"Điều\s+\d+", r["answer"]), (
                f"Missing Điều X in answer for Q{r['id']}"
            )

        # Validate
        valid, errors = validate_submission(results)
        assert valid is True, f"Validation failed: {errors}"

        # Package
        zip_path = save_submission(results, tmp_path)
        import zipfile as zf
        with zf.ZipFile(zip_path) as z:
            assert z.namelist() == ["results.json"]
            data = json.loads(z.read("results.json").decode("utf-8"))
        assert len(data) == 3
