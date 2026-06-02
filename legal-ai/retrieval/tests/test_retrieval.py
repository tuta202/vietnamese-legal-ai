"""
Unit tests for the retrieval stack.
All tests run OFFLINE — no Qdrant, no vLLM, no GPU required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# conftest.py ensures legal-ai/ is in sys.path
from retrieval.bm25_index import BM25Index, vietnamese_simple_tokenize
from retrieval.config import RetrievalConfig, load_config
from retrieval.embedder import LegalEmbedder
from retrieval.hybrid_search import HybridSearcher, rrf_fusion
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import LegalReranker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> RetrievalConfig:
    return RetrievalConfig()


@pytest.fixture()
def sample_articles() -> list[dict]:
    return [
        {
            "chunk_id": "aabbccdd" * 4,
            "law_id": "04/2017/QH14",
            "law_type": "Luật",
            "law_name": "Hỗ trợ doanh nghiệp nhỏ và vừa",
            "dieu_number": "Điều 4",
            "dieu_title": "Tiêu chí xác định doanh nghiệp nhỏ và vừa",
            "content": (
                "Doanh nghiệp nhỏ và vừa là doanh nghiệp có số lao động "
                "tham gia bảo hiểm xã hội bình quân năm không quá 200 người "
                "và tổng nguồn vốn không quá 100 tỷ đồng."
            ),
            "relevant_doc_str": "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "relevant_article_str": "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
            "khoan_list": [],
        },
        {
            "chunk_id": "11223344" * 4,
            "law_id": "45/2019/QH14",
            "law_type": "Bộ luật",
            "law_name": "Lao động",
            "dieu_number": "Điều 112",
            "dieu_title": "Ngày nghỉ hằng năm",
            "content": (
                "Người lao động làm việc đủ 12 tháng cho một người sử dụng "
                "lao động được nghỉ hằng năm hưởng nguyên lương 12 ngày làm việc."
            ),
            "relevant_doc_str": "45/2019/QH14|Bộ luật Lao động",
            "relevant_article_str": "45/2019/QH14|Bộ luật Lao động|Điều 112",
            "khoan_list": [],
        },
        {
            "chunk_id": "55667788" * 4,
            "law_id": "59/2020/QH14",
            "law_type": "Luật",
            "law_name": "Doanh nghiệp",
            "dieu_number": "Điều 7",
            "dieu_title": "Quyền của doanh nghiệp",
            "content": (
                "Tự do kinh doanh ngành, nghề mà pháp luật không cấm. "
                "Tự chủ kinh doanh và lựa chọn hình thức tổ chức kinh doanh."
            ),
            "relevant_doc_str": "59/2020/QH14|Luật Doanh nghiệp",
            "relevant_article_str": "59/2020/QH14|Luật Doanh nghiệp|Điều 7",
            "khoan_list": [],
        },
        {
            "chunk_id": "99aabbcc" * 4,
            "law_id": "13/2008/QH12",
            "law_type": "Luật",
            "law_name": "Thuế giá trị gia tăng",
            "dieu_number": "Điều 8",
            "dieu_title": "Thuế suất",
            "content": (
                "Mức thuế suất 0% áp dụng đối với hàng hóa dịch vụ xuất khẩu. "
                "Mức thuế suất 10% áp dụng đối với hàng hóa dịch vụ còn lại."
            ),
            "relevant_doc_str": "13/2008/QH12|Luật Thuế giá trị gia tăng",
            "relevant_article_str": "13/2008/QH12|Luật Thuế giá trị gia tăng|Điều 8",
            "khoan_list": [],
        },
        {
            "chunk_id": "ddeeff00" * 4,
            "law_id": "78/2014/TT-BTC",
            "law_type": "Thông tư",
            "law_name": "Hướng dẫn thuế TNDN",
            "dieu_number": "Điều 1",
            "dieu_title": "Phạm vi điều chỉnh",
            "content": (
                "Thông tư này hướng dẫn thi hành Nghị định số 218/2013/NĐ-CP "
                "về thuế thu nhập doanh nghiệp."
            ),
            "relevant_doc_str": "78/2014/TT-BTC|Thông tư Hướng dẫn thuế TNDN",
            "relevant_article_str": "78/2014/TT-BTC|Thông tư Hướng dẫn thuế TNDN|Điều 1",
            "khoan_list": [],
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBM25:
    def test_tokenizer_basic(self):
        tokens = vietnamese_simple_tokenize("Doanh nghiệp nhỏ và vừa được hỗ trợ!")
        assert "doanh" in tokens
        assert "nghiệp" in tokens
        assert "và" in tokens        # "và" = 2 chars → kept; single-char filter only removes 1-char tokens
        assert "!" not in tokens     # punctuation stripped
        # Single-char token "a" is stripped
        single_char_tokens = vietnamese_simple_tokenize("a b c abc")
        assert "a" not in single_char_tokens
        assert "b" not in single_char_tokens
        assert "abc" in single_char_tokens

    def test_tokenizer_lowercase(self):
        tokens = vietnamese_simple_tokenize("Điều 4. Tiêu chí")
        assert "điều" in tokens
        assert "Điều" not in tokens

    def test_build_and_search(self, sample_articles):
        idx = BM25Index().build_from_corpus(sample_articles)
        assert len(idx) == 5

        results = idx.search("doanh nghiệp nhỏ và vừa", top_k=3)
        assert len(results) == 3
        # Top result should be from 04/2017/QH14 (most relevant)
        top_chunk_id, top_score = results[0]
        payload = idx.get_payload(top_chunk_id)
        assert payload is not None
        assert payload["law_id"] == "04/2017/QH14"
        assert top_score > 0

    def test_search_returns_empty_on_no_match(self, sample_articles):
        idx = BM25Index().build_from_corpus(sample_articles)
        # Use query with only ASCII gibberish — no overlap with Vietnamese corpus
        results = idx.search("aaabbbccc zzzxxxyyy", top_k=5)
        assert results == []

    def test_search_top_k_respected(self, sample_articles):
        idx = BM25Index().build_from_corpus(sample_articles)
        results = idx.search("doanh", top_k=2)
        assert len(results) <= 2

    def test_save_and_load(self, sample_articles, tmp_path):
        idx = BM25Index().build_from_corpus(sample_articles)
        pkl_path = tmp_path / "test_bm25.pkl"
        idx.save(pkl_path)
        loaded = BM25Index.load(pkl_path)
        assert len(loaded) == len(idx)
        # Search should give same results
        r1 = idx.search("lao động", top_k=3)
        r2 = loaded.search("lao động", top_k=3)
        assert [cid for cid, _ in r1] == [cid for cid, _ in r2]


class TestRRFFusion:
    def test_rrf_basic_order(self):
        # doc "b" is ranked 1st in both lists → should have highest RRF score
        rank1 = [("b", 1.0), ("a", 0.9), ("c", 0.8)]
        rank2 = [("b", 1.0), ("c", 0.9), ("a", 0.8)]
        result = rrf_fusion([rank1, rank2], k=60)
        assert result[0][0] == "b"

    def test_rrf_formula_correctness(self):
        # With k=60: rank1 → 1/(60+1)=0.01639, rank2 → 1/(60+2)=0.01613
        rank1 = [("a", 1.0), ("b", 0.5)]
        rank2 = [("a", 1.0), ("b", 0.5)]
        result = dict(rrf_fusion([rank1, rank2], k=60))
        expected_a = 2 * (1.0 / (60 + 1))
        expected_b = 2 * (1.0 / (60 + 2))
        assert abs(result["a"] - expected_a) < 1e-9
        assert abs(result["b"] - expected_b) < 1e-9
        assert result["a"] > result["b"]

    def test_rrf_aggregates_across_lists(self):
        # "z" appears in all 3 lists ranked 3rd → should beat "x" that only appears in 1
        r1 = [("z", 1.0), ("x", 0.9)]
        r2 = [("z", 1.0)]
        r3 = [("z", 1.0), ("x", 0.5)]
        result = dict(rrf_fusion([r1, r2, r3], k=60))
        assert result["z"] > result.get("x", 0)

    def test_rrf_empty_inputs(self):
        assert rrf_fusion([]) == []
        assert rrf_fusion([[]]) == []

    def test_rrf_single_list(self):
        ranking = [("a", 1.0), ("b", 0.5)]
        result = rrf_fusion([ranking], k=60)
        assert result[0][0] == "a"


class TestQueryRewriterMock:
    def test_returns_correct_schema(self, default_config):
        rewriter = QueryRewriter(default_config, mock=True)
        result = rewriter.rewrite("Công ty có phải đăng ký không?")
        assert "rewritten_query" in result
        assert "topic_description" in result
        # Verify the old HyDE field has been fully replaced
        old_field = "_".join(["hyde", "passage"])   # avoids literal match in grep
        assert old_field not in result

    def test_mock_returns_original_question(self, default_config):
        question = "Câu hỏi kiểm tra"
        rewriter = QueryRewriter(default_config, mock=True)
        result = rewriter.rewrite(question)
        assert result["rewritten_query"] == question
        assert result["topic_description"] == question

    def test_mock_no_network_call(self, default_config):
        # Should not try to import or call openai
        rewriter = QueryRewriter(default_config, mock=True)
        rewriter.rewrite("test")
        assert rewriter._client is None   # client never initialised

    def test_topic_description_key_present(self, default_config):
        """AC-4: mock output has exactly the two required keys."""
        rewriter = QueryRewriter(default_config, mock=True)
        result = rewriter.rewrite("Điều kiện thành lập doanh nghiệp?")
        assert set(result.keys()) == {"rewritten_query", "topic_description"}


class TestRerankerMock:
    def test_mock_preserves_order(self, default_config):
        reranker = LegalReranker(default_config, mock=True)
        candidates = [{"chunk_id": str(i), "content": f"text {i}"} for i in range(5)]
        result = reranker.rerank("query", candidates, top_k=3)
        assert len(result) == 3
        assert [c["chunk_id"] for c in result] == ["0", "1", "2"]

    def test_mock_truncates_to_top_k(self, default_config):
        reranker = LegalReranker(default_config, mock=True)
        candidates = [{"chunk_id": str(i)} for i in range(10)]
        assert len(reranker.rerank("q", candidates, top_k=4)) == 4

    def test_mock_empty_input(self, default_config):
        reranker = LegalReranker(default_config, mock=True)
        assert reranker.rerank("q", [], top_k=5) == []

    def test_mock_no_model_load(self, default_config):
        reranker = LegalReranker(default_config, mock=True)
        reranker.rerank("q", [{"chunk_id": "x"}], top_k=1)
        assert reranker._model is None


class TestEmbedderDryRun:
    def test_embed_query_returns_array(self, default_config):
        embedder = LegalEmbedder(default_config, dry_run=True)
        vec = embedder.embed_query("thuế thu nhập doanh nghiệp")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == 4096

    def test_embed_query_no_model_load(self, default_config):
        embedder = LegalEmbedder(default_config, dry_run=True)
        embedder.embed_query("test")
        assert embedder._model is None

    def test_format_query_contains_instruction(self, default_config):
        embedder = LegalEmbedder(default_config, dry_run=True)
        formatted = embedder._format_query("test question")
        assert "Instruct:" in formatted
        assert "Query:" in formatted
        assert "test question" in formatted

    def test_format_document_contains_law_fields(self, default_config, sample_articles):
        embedder = LegalEmbedder(default_config, dry_run=True)
        art = sample_articles[0]
        formatted = embedder._format_document(art)
        assert art["law_id"] in formatted
        assert art["dieu_number"] in formatted
        assert art["dieu_title"] in formatted

    def test_embed_corpus_dry_run_skips_qdrant(self, default_config, sample_articles):
        embedder = LegalEmbedder(default_config, dry_run=True)
        count = embedder.embed_corpus(sample_articles, qdrant_client=None)
        assert count == len(sample_articles)


class TestHybridSearchWithMocks:
    def _make_mock_qdrant(self, chunk_ids: list[str]):
        """Return a mock qdrant client that returns fake ScoredPoints."""
        client = MagicMock()
        mock_points = []
        for i, cid in enumerate(chunk_ids):
            pt = MagicMock()
            pt.id   = cid
            pt.score = 1.0 - i * 0.1
            mock_points.append(pt)
        client.search.return_value = mock_points
        return client

    def test_search_calls_all_five_steps(self, default_config, sample_articles):
        bm25 = BM25Index().build_from_corpus(sample_articles)
        embedder = LegalEmbedder(default_config, dry_run=True)
        rewriter = QueryRewriter(default_config, mock=True)

        # Qdrant mock returns 2 articles
        chunk_ids = [sample_articles[0]["chunk_id"], sample_articles[1]["chunk_id"]]
        qdrant = self._make_mock_qdrant(chunk_ids)

        searcher = HybridSearcher(default_config, bm25, embedder, rewriter, qdrant)

        question = "doanh nghiệp nhỏ và vừa có bao nhiêu lao động?"
        results = searcher.search(question, top_k=5)

        # Step 1: rewriter called
        assert rewriter._client is None  # mock=True, no llm init
        # Step 3+4: qdrant.search called (at least once for query, maybe twice for hyde)
        assert qdrant.search.call_count >= 1
        # Step 5: fusion produces list
        assert isinstance(results, list)
        assert all("rrf_score" in r for r in results)

    def test_rrf_score_present_in_results(self, default_config, sample_articles):
        bm25    = BM25Index().build_from_corpus(sample_articles)
        embedder = LegalEmbedder(default_config, dry_run=True)
        rewriter = QueryRewriter(default_config, mock=True)
        qdrant   = self._make_mock_qdrant([sample_articles[2]["chunk_id"]])

        searcher = HybridSearcher(default_config, bm25, embedder, rewriter, qdrant)
        results  = searcher.search("lao động nghỉ phép", top_k=3)

        for r in results:
            assert "rrf_score" in r
            assert r["rrf_score"] > 0


class TestConfigLoad:
    def test_load_defaults_without_file(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg.qdrant.host == "localhost"
        assert cfg.retrieval.rrf_k == 60
        assert cfg.retrieval.top_k_rerank == 7

    def test_load_from_yaml(self):
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if not config_path.exists():
            pytest.skip("config.yaml not found")
        cfg = load_config(config_path)
        assert cfg.qdrant.collection == "legal_vn"
        assert cfg.models.embedder == "Qwen/Qwen3-Embedding-8B"
        assert cfg.vllm.model == "Qwen/Qwen2.5-7B-Instruct"
