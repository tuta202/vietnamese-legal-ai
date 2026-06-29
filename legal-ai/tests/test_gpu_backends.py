"""
Offline tests for the pure self-deployed GPU endpoints backend (gpu_backends.py).

All run with mock/dry_run=True — they must never construct an OpenAI client or
touch google.auth (lazy init), and must inherit the base components' offline
behaviour unchanged. Also covers GpuConfig loading + validation and the
all-Gpu factory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpu_backends import (
    GpuEmbedder,
    GpuGenerator,
    GpuQueryRewriter,
    GpuReranker,
    make_gpu_components,
)
from llm_candidate_verifier import VerifierWorker
from intent_ranked_hits_clean import WorkerContext
from retrieval.config import (
    EmbeddingConfig,
    RetrievalConfig,
    load_config,
    validate_config,
)

_CONFIG_GPU = Path(__file__).resolve().parent.parent / "config_gpu_clean.yaml"


@pytest.fixture
def gpu_cfg() -> RetrievalConfig:
    """A complete, valid pure-Gpu config."""
    cfg = RetrievalConfig(backend="gpu")
    cfg.embedding = EmbeddingConfig(dimension=4096)
    cfg.qdrant.vector_size = 4096
    cfg.qdrant.host = "localhost"
    cfg.qdrant.collection = "legal_vn_garden"
    cfg.gpu.embed_endpoint_id = "111"
    cfg.gpu.embed_dns = "111.region-proj.prediction.vertexai.goog"
    cfg.gpu.llm_endpoint_id = "222"
    cfg.gpu.project_id = "proj"
    return cfg


_CANDIDATES = [
    {"chunk_id": "a", "dieu_number": "Điều 1", "dieu_title": "T1", "content": "c1"},
    {"chunk_id": "b", "dieu_number": "Điều 2", "dieu_title": "T2", "content": "c2"},
    {"chunk_id": "c", "dieu_number": "Điều 3", "dieu_title": "T3", "content": "c3"},
]


class TestLazyInit:
    def test_no_client_built_in_mock(self, gpu_cfg):
        rw = GpuQueryRewriter(gpu_cfg, mock=True)
        gen = GpuGenerator(gpu_cfg, mock=True)
        emb = GpuEmbedder(gpu_cfg, dry_run=True)
        rr = GpuReranker(gpu_cfg, mock=True)
        assert rw._client is None
        assert gen._client is None
        assert emb._model is None
        assert rr._model is None


class TestInheritedMockBehaviour:
    def test_rewriter_echoes_in_mock(self, gpu_cfg):
        out = GpuQueryRewriter(gpu_cfg, mock=True).rewrite("câu hỏi")
        assert out["rewritten_query"] == "câu hỏi"

    def test_reranker_truncates_in_mock(self, gpu_cfg):
        out = GpuReranker(gpu_cfg, mock=True).rerank("q", _CANDIDATES, top_k=2)
        assert len(out) == 2

    def test_reranker_empty(self, gpu_cfg):
        assert GpuReranker(gpu_cfg, mock=True).rerank("q", [], top_k=3) == []

    def test_generator_mock_has_dieu_and_disclaimer(self, gpu_cfg):
        ans = GpuGenerator(gpu_cfg, mock=True).generate("q", _CANDIDATES)
        assert "Điều 1" in ans
        assert "Lưu ý:" in ans

    def test_embedder_dry_run_dim(self, gpu_cfg):
        vec = GpuEmbedder(gpu_cfg, dry_run=True).embed_query("x")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 4096


class TestFactory:
    def test_factory_is_all_gpu(self, gpu_cfg):
        emb, rw, rr, gen = make_gpu_components(gpu_cfg, mock=True)
        assert isinstance(emb, GpuEmbedder)
        assert isinstance(rw, GpuQueryRewriter)
        assert isinstance(rr, GpuReranker)
        assert isinstance(gen, GpuGenerator)


class TestValidation:
    def test_valid_gpu_config_passes(self, gpu_cfg):
        assert validate_config(gpu_cfg) == []

    def test_missing_endpoints_flagged(self):
        cfg = RetrievalConfig(backend="gpu")
        cfg.embedding = EmbeddingConfig(dimension=4096)
        cfg.qdrant.vector_size = 4096
        errs = validate_config(cfg)
        assert any("embed_endpoint_id" in e for e in errs)
        assert any("llm_endpoint_id" in e for e in errs)
        assert any("project_id" in e for e in errs)

    def test_dimension_vector_size_mismatch_flagged(self, gpu_cfg):
        gpu_cfg.qdrant.vector_size = 3072   # != embedding.dimension (4096)
        errs = validate_config(gpu_cfg)
        assert any("vector_size" in e for e in errs)


class TestConfigFileLoads:
    def test_config_gpu_yaml_loads(self):
        cfg = load_config(_CONFIG_GPU)
        assert cfg.backend == "gpu"
        assert cfg.gpu.embed_model == "Qwen3-Embedding-8B"
        assert cfg.gpu.llm_model == "google/gemma-3-12b-it"
        assert cfg.gpu.embed_region == "asia-northeast1"
        assert cfg.gpu.region == "asia-southeast1"
        assert cfg.embedding.dimension == 4096
        assert cfg.qdrant.vector_size == 4096
        assert cfg.qdrant.collection == "legal_vn_qwen3_asof_20260301_v1"
        assert cfg.retrieval.top_k_rerank == 6
        assert cfg.retrieval.top_k_fusion == 200

    def test_clean_gpu_config_uses_qwen_collection(self):
        cfg = load_config(_CONFIG_GPU)
        assert cfg.backend == "gpu"
        assert cfg.models.embedder == "Qwen3-Embedding-8B"
        assert cfg.models.reranker == "BAAI/bge-reranker-v2-m3"
        assert cfg.embedding.dimension == 4096
        assert cfg.qdrant.vector_size == 4096
        assert cfg.qdrant.collection == "legal_vn_qwen3_asof_20260301_v1"


class TestGpuVerifier:
    def test_verifier_uses_gemma_adapter(self, gpu_cfg, monkeypatch):
        calls = []

        def fake_chat(config, *, system, user, temperature, max_tokens):
            calls.append(
                {
                    "config": config,
                    "system": system,
                    "user": user,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return '{"selected_article_keys":["A1"],"confidence":"high"}'

        monkeypatch.setattr("gpu_backends.gpu_chat_complete", fake_chat)
        worker = VerifierWorker(gpu_cfg)
        raw = worker.call(
            "question",
            [{"key": "A1", "content": "article"}],
            system_prompt="system",
            legal_intents=["intent"],
        )

        assert "A1" in raw
        assert len(calls) == 1
        assert calls[0]["config"] is gpu_cfg
        assert '"legal_intents"' in calls[0]["user"]
        assert calls[0]["temperature"] == 0.0
        assert calls[0]["max_tokens"] == 2048


class TestGpuIntentRetrievalRouting:
    def test_worker_context_uses_gpu_embedder(self, gpu_cfg, monkeypatch):
        marker = object()

        monkeypatch.setattr(
            "gpu_backends.make_gpu_components",
            lambda config, mock: (marker, object(), object(), object()),
        )
        monkeypatch.setattr(
            "qdrant_client.QdrantClient",
            lambda *args, **kwargs: object(),
        )

        context = WorkerContext(gpu_cfg, bm25=object())
        assert context.embedder is marker
