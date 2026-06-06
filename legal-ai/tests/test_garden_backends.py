"""
Offline tests for the pure Vertex AI Model Garden backend (garden_backends.py).

All run with mock/dry_run=True — they must never construct an OpenAI client or
touch google.auth (lazy init), and must inherit the base components' offline
behaviour unchanged. Also covers GardenConfig loading + validation and the
all-Garden factory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from garden_backends import (
    GardenEmbedder,
    GardenGenerator,
    GardenQueryRewriter,
    GardenReranker,
    make_garden_components,
)
from retrieval.config import (
    EmbeddingConfig,
    RetrievalConfig,
    load_config,
    validate_config,
)

_CONFIG_GARDEN = Path(__file__).resolve().parent.parent / "config_garden.yaml"


@pytest.fixture
def garden_cfg() -> RetrievalConfig:
    """A complete, valid pure-Garden config."""
    cfg = RetrievalConfig(backend="garden")
    cfg.embedding = EmbeddingConfig(dimension=4096)
    cfg.qdrant.vector_size = 4096
    cfg.qdrant.host = "localhost"
    cfg.qdrant.collection = "legal_vn_garden"
    cfg.garden.embed_endpoint_id = "111"
    cfg.garden.llm_endpoint_id = "222"
    cfg.garden.project_id = "proj"
    return cfg


_CANDIDATES = [
    {"chunk_id": "a", "dieu_number": "Điều 1", "dieu_title": "T1", "content": "c1"},
    {"chunk_id": "b", "dieu_number": "Điều 2", "dieu_title": "T2", "content": "c2"},
    {"chunk_id": "c", "dieu_number": "Điều 3", "dieu_title": "T3", "content": "c3"},
]


class TestLazyInit:
    def test_no_client_built_in_mock(self, garden_cfg):
        rw = GardenQueryRewriter(garden_cfg, mock=True)
        gen = GardenGenerator(garden_cfg, mock=True)
        emb = GardenEmbedder(garden_cfg, dry_run=True)
        rr = GardenReranker(garden_cfg, mock=True)
        assert rw._client is None
        assert gen._client is None
        assert emb._model is None
        assert rr._model is None


class TestInheritedMockBehaviour:
    def test_rewriter_echoes_in_mock(self, garden_cfg):
        out = GardenQueryRewriter(garden_cfg, mock=True).rewrite("câu hỏi")
        assert out["rewritten_query"] == "câu hỏi"

    def test_reranker_truncates_in_mock(self, garden_cfg):
        out = GardenReranker(garden_cfg, mock=True).rerank("q", _CANDIDATES, top_k=2)
        assert len(out) == 2

    def test_reranker_empty(self, garden_cfg):
        assert GardenReranker(garden_cfg, mock=True).rerank("q", [], top_k=3) == []

    def test_generator_mock_has_dieu_and_disclaimer(self, garden_cfg):
        ans = GardenGenerator(garden_cfg, mock=True).generate("q", _CANDIDATES)
        assert "Điều 1" in ans
        assert "Lưu ý:" in ans

    def test_embedder_dry_run_dim(self, garden_cfg):
        vec = GardenEmbedder(garden_cfg, dry_run=True).embed_query("x")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 4096


class TestFactory:
    def test_factory_is_all_garden(self, garden_cfg):
        emb, rw, rr, gen = make_garden_components(garden_cfg, mock=True)
        assert isinstance(emb, GardenEmbedder)
        assert isinstance(rw, GardenQueryRewriter)
        assert isinstance(rr, GardenReranker)
        assert isinstance(gen, GardenGenerator)


class TestValidation:
    def test_valid_garden_config_passes(self, garden_cfg):
        assert validate_config(garden_cfg) == []

    def test_missing_endpoints_flagged(self):
        cfg = RetrievalConfig(backend="garden")
        cfg.embedding = EmbeddingConfig(dimension=4096)
        cfg.qdrant.vector_size = 4096
        errs = validate_config(cfg)
        assert any("embed_endpoint_id" in e for e in errs)
        assert any("llm_endpoint_id" in e for e in errs)
        assert any("project_id" in e for e in errs)

    def test_dimension_vector_size_mismatch_flagged(self, garden_cfg):
        garden_cfg.qdrant.vector_size = 3072   # != embedding.dimension (4096)
        errs = validate_config(garden_cfg)
        assert any("vector_size" in e for e in errs)


class TestConfigFileLoads:
    def test_config_garden_yaml_loads(self):
        cfg = load_config(_CONFIG_GARDEN)
        assert cfg.backend == "garden"
        assert cfg.garden.embed_model == "Qwen3-Embedding-8B"
        assert cfg.garden.llm_model == "gemma-3-12b-it"
        assert cfg.embedding.dimension == 4096
        assert cfg.qdrant.vector_size == 4096
        assert cfg.qdrant.collection == "legal_vn_garden"
        assert cfg.retrieval.top_k_rerank == 5
