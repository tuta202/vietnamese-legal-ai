"""
Production-readiness tests for the pipeline:
  • per-question error isolation (one failure never aborts the batch)
  • fail-fast config validation on a real run
  • dense-resolution regression (TIP-VERTEX-003): dense hits keyed by the payload's
    raw chunk_id, not the dashed Qdrant point UUID.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import LegalAIPipeline


@pytest.fixture
def mock_pipeline() -> LegalAIPipeline:
    return LegalAIPipeline(mock=True)


class TestErrorIsolation:
    def test_failing_step_yields_fallback_entry(self, mock_pipeline, monkeypatch):
        def boom(_state):
            raise RuntimeError("simulated backend failure")
        monkeypatch.setattr(mock_pipeline, "step_retrieve", boom)

        entry = mock_pipeline.process_question(99, "câu hỏi pháp lý")
        assert set(entry) == {
            "id", "question", "answer", "relevant_docs", "relevant_articles"
        }
        assert entry["id"] == 99
        assert entry["relevant_articles"] == []
        assert "Lưu ý:" in entry["answer"]  # disclaimer fallback

    def test_batch_continues_past_failure(self, mock_pipeline, monkeypatch):
        calls = {"n": 0}
        real = mock_pipeline.step_generate

        def flaky(state):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom on first question")
            return real(state)
        monkeypatch.setattr(mock_pipeline, "step_generate", flaky)

        out = mock_pipeline.process_questions(
            [{"id": 1, "question": "q1"}, {"id": 2, "question": "q2"}]
        )
        assert len(out) == 2          # batch not aborted
        assert out[0]["relevant_articles"] == []   # first failed → fallback


class TestFailFastConfig:
    def test_empty_vertex_creds_raise(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("backend: vertex_ai\n", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            LegalAIPipeline(config_path=p, mock=False)
        assert "qdrant.url" in str(exc.value)

    def test_mock_skips_validation(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("backend: vertex_ai\n", encoding="utf-8")
        # mock=True must not validate / must not raise
        LegalAIPipeline(config_path=p, mock=True)


class TestDenseResolutionRegression:
    def test_dense_keyed_by_payload_chunk_id(self, mock_pipeline):
        # Qdrant returns a point whose .id is a dashed UUID but whose payload
        # carries the raw md5 chunk_id — the fix must key on the payload.
        pt = SimpleNamespace(
            id="6f1e2c3d-aaaa-bbbb-cccc-000000000000",
            score=0.93,
            payload={
                "chunk_id": "rawmd5hex",
                "law_id": "59/2020/QH14", "law_type": "Luật",
                "law_name": "Doanh nghiệp", "dieu_number": "Điều 5",
                "dieu_title": "Tiêu đề", "content": "nội dung",
                "relevant_doc_str": "59/2020/QH14|Luật Doanh nghiệp",
                "relevant_article_str": "59/2020/QH14|Luật Doanh nghiệp|Điều 5",
            },
        )
        mock_pipeline._qdrant = SimpleNamespace(
            query_points=lambda **kw: SimpleNamespace(points=[pt])
        )
        hits, payloads = mock_pipeline._dense_search(
            np.zeros(mock_pipeline.config.embedding.dimension, dtype=np.float32)
        )
        assert hits == [("rawmd5hex", pytest.approx(0.93))]
        assert "rawmd5hex" in payloads

        resolved = mock_pipeline._resolve_payloads(hits, payloads)
        assert resolved[0]["relevant_article_str"] == \
            "59/2020/QH14|Luật Doanh nghiệp|Điều 5"

    def test_normalize_payload_builds_strings_from_article_fields(self):
        qp = {
            "law_id": "04/2017/QH14", "law_type": "Luật", "law_name": "Hỗ trợ DNNVV",
            "article_number": "Điều 4", "article_title": "Tiêu chí", "content": "x",
        }
        out = LegalAIPipeline._normalize_payload(qp, "cid")
        assert out["dieu_number"] == "Điều 4"
        assert out["relevant_article_str"] == "04/2017/QH14|Luật Hỗ trợ DNNVV|Điều 4"
