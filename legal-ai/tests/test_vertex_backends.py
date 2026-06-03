"""
Offline tests for the Vertex AI subclasses (vertex_backends.py).

All run with mock/dry_run=True — they must never construct a google-genai client
(lazy init) and must inherit the base components' offline behaviour unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.config import EmbeddingConfig, RetrievalConfig
from vertex_backends import (
    VertexEmbedder,
    VertexGenerator,
    VertexQueryRewriter,
    VertexReranker,
)


@pytest.fixture
def vertex_cfg() -> RetrievalConfig:
    cfg = RetrievalConfig(backend="vertex_ai")
    cfg.embedding = EmbeddingConfig(dimension=3072)
    cfg.models.embedder = "gemini-embedding-001"
    cfg.models.reranker = "gemini-2.5-flash-lite"
    cfg.vllm.model = "gemini-2.5-flash-lite"
    return cfg


_CANDIDATES = [
    {"chunk_id": "a", "dieu_number": "Điều 1", "dieu_title": "T1", "content": "c1"},
    {"chunk_id": "b", "dieu_number": "Điều 2", "dieu_title": "T2", "content": "c2"},
    {"chunk_id": "c", "dieu_number": "Điều 3", "dieu_title": "T3", "content": "c3"},
]


class TestLazyInit:
    def test_no_client_built_in_mock(self, vertex_cfg):
        rw = VertexQueryRewriter(vertex_cfg, mock=True)
        gen = VertexGenerator(vertex_cfg, mock=True)
        emb = VertexEmbedder(vertex_cfg, dry_run=True)
        rr = VertexReranker(vertex_cfg, mock=True)
        assert rw._client is None
        assert gen._client is None
        assert emb._genai is None and emb._model is None
        assert rr._genai is None and rr._model is None


class TestInheritedMockBehaviour:
    def test_rewriter_echoes_in_mock(self, vertex_cfg):
        out = VertexQueryRewriter(vertex_cfg, mock=True).rewrite("câu hỏi")
        assert out["rewritten_query"] == "câu hỏi"

    def test_reranker_truncates_in_mock(self, vertex_cfg):
        out = VertexReranker(vertex_cfg, mock=True).rerank("q", _CANDIDATES, top_k=2)
        assert len(out) == 2

    def test_reranker_empty(self, vertex_cfg):
        assert VertexReranker(vertex_cfg, mock=True).rerank("q", [], top_k=3) == []

    def test_generator_mock_has_dieu_and_disclaimer(self, vertex_cfg):
        ans = VertexGenerator(vertex_cfg, mock=True).generate("q", _CANDIDATES)
        assert "Điều 1" in ans
        assert "Lưu ý:" in ans

    def test_embedder_dry_run_dim(self, vertex_cfg):
        vec = VertexEmbedder(vertex_cfg, dry_run=True).embed_query("x")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 3072


class TestRerankScoreParsing:
    def test_parse_plain_json(self):
        out = VertexReranker._parse_scores('[{"index": 0, "score": 8}, {"index": 1, "score": 3}]')
        assert out == {0: 8.0, 1: 3.0}

    def test_parse_with_code_fence_and_prose(self):
        raw = 'Kết quả:\n```json\n[{"index": 2, "score": 10}]\n```'
        assert VertexReranker._parse_scores(raw) == {2: 10.0}

    def test_parse_garbage_returns_none(self):
        assert VertexReranker._parse_scores("not json at all") is None
