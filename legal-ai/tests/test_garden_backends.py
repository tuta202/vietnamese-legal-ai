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

from backends_common import parse_rerank_scores
from garden_backends import (
    GardenCoTReranker,
    GardenCrossEncoderReranker,
    GardenEmbedder,
    GardenGenerator,
    GardenQueryRewriter,
    GardenReranker,
    GardenTwoTierReranker,
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
    cfg.garden.embed_dns = "111.region-proj.prediction.vertexai.goog"
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
        # reranker is now the two-tier composite (BGE cross-encoder → Gemma CoT)
        assert isinstance(rr, GardenTwoTierReranker)
        assert isinstance(rr.ce, GardenCrossEncoderReranker)
        assert isinstance(rr.cot, GardenCoTReranker)
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

    def test_funnel_constraints_flagged(self, garden_cfg):
        garden_cfg.retrieval.top_k_fusion = 10
        garden_cfg.retrieval.top_k_rerank_ce = 20   # > top_k_fusion
        garden_cfg.retrieval.top_k_rerank = 25      # > top_k_rerank_ce
        errs = validate_config(garden_cfg)
        assert any("top_k_rerank_ce" in e for e in errs)
        assert any("top_k_rerank " in e for e in errs)

    def test_empty_query_instruction_flagged(self, garden_cfg):
        garden_cfg.embedding.query_instruction = ""
        errs = validate_config(garden_cfg)
        assert any("query_instruction" in e for e in errs)

    def test_ce_endpoint_not_required_for_load(self, garden_cfg):
        # Build-only: BGE endpoint may be undeployed (ce_* empty) yet config valid.
        garden_cfg.garden.ce_endpoint_id = ""
        garden_cfg.garden.ce_dns = ""
        assert validate_config(garden_cfg) == []


# ---------------------------------------------------------------------------
# TIP-013 — Qwen3 query instruction (asymmetric query/document)
# ---------------------------------------------------------------------------

class TestQueryInstruction:
    def test_query_has_instruct_prefix(self, garden_cfg):
        formatted = GardenEmbedder(garden_cfg, dry_run=True)._format_query("test")
        assert formatted.startswith("Instruct: ")
        assert "\nQuery: test" in formatted
        assert garden_cfg.embedding.query_instruction in formatted

    def test_document_has_no_instruct_prefix(self, garden_cfg):
        art = {"law_id": "L", "law_type": "Luật", "law_name": "X",
               "dieu_number": "Điều 1", "dieu_title": "T", "content": "c"}
        doc = GardenEmbedder(garden_cfg, dry_run=True)._format_document(art)
        assert "Instruct:" not in doc
        assert "Query:" not in doc

    def test_query_instruction_is_config_driven(self, garden_cfg):
        garden_cfg.embedding.query_instruction = "CUSTOM_INSTRUCTION"
        out = GardenEmbedder(garden_cfg, dry_run=True)._format_query("abc")
        assert "CUSTOM_INSTRUCTION" in out


# ---------------------------------------------------------------------------
# TIP-013 — TIER 1 cross-encoder reranker
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    def test_mock_truncates(self, garden_cfg):
        ce = GardenCrossEncoderReranker(garden_cfg, mock=True)
        assert len(ce.rerank("q", _CANDIDATES, top_k=2)) == 2

    def test_sorts_attaches_ce_score_and_uses_original_query(self, garden_cfg, monkeypatch):
        ce = GardenCrossEncoderReranker(garden_cfg, mock=False)
        seen = {}

        def fake_scores(question, candidates):
            seen["q"] = question
            return [0.1, 0.9, 0.5]   # candidate "b" highest, then "c", then "a"

        monkeypatch.setattr(ce, "_ce_scores", fake_scores)
        out = ce.rerank("ORIGINAL_QUERY", _CANDIDATES, top_k=2)
        assert seen["q"] == "ORIGINAL_QUERY"          # original query, not rewritten
        assert [d["chunk_id"] for d in out] == ["b", "c"]
        assert out[0]["ce_score"] == 0.9
        assert all("ce_score" in d for d in out)


# ---------------------------------------------------------------------------
# TIP-013 — TIER 2 LLM CoT reranker + reasoning-tolerant parse
# ---------------------------------------------------------------------------

class TestCoTReranker:
    def test_mock_truncates(self, garden_cfg):
        cot = GardenCoTReranker(garden_cfg, mock=True)
        assert len(cot.rerank("q", _CANDIDATES, top_k=2)) == 2

    def test_scores_sort_via_llm(self, garden_cfg, monkeypatch):
        cot = GardenCoTReranker(garden_cfg, mock=False)
        monkeypatch.setattr(cot, "_llm_scores", lambda q, c: {0: 1.0, 1: 9.0, 2: 5.0})
        out = cot.rerank("q", _CANDIDATES, top_k=2)
        assert [d["chunk_id"] for d in out] == ["b", "c"]

    def test_parse_fail_preserves_tier1_order(self, garden_cfg, monkeypatch):
        cot = GardenCoTReranker(garden_cfg, mock=False)
        monkeypatch.setattr(cot, "_llm_scores", lambda q, c: None)
        out = cot.rerank("q", _CANDIDATES, top_k=3)
        assert [d["chunk_id"] for d in out] == ["a", "b", "c"]   # input order kept


class TestParseReasoning:
    def test_parses_json_array_after_reasoning(self):
        raw = (
            "Suy luận:\n1. Yếu tố cốt lõi... liên quan [0] và [1].\n"
            "2. Điều [2] dư thừa.\n"
            'Kết luận:\n[{"index": 0, "score": 9}, {"index": 2, "score": 4}]'
        )
        assert parse_rerank_scores(raw) == {0: 9.0, 2: 4.0}

    def test_plain_array_still_parses(self):
        assert parse_rerank_scores('[{"index": 0, "score": 8}]') == {0: 8.0}

    def test_garbage_returns_none(self):
        assert parse_rerank_scores("not json at all") is None


# ---------------------------------------------------------------------------
# TIP-013 — composite two-tier funnel
# ---------------------------------------------------------------------------

class TestTwoTier:
    def test_composite_funnels_ce_then_cot(self, garden_cfg, monkeypatch):
        garden_cfg.retrieval.top_k_rerank_ce = 2
        rr = GardenTwoTierReranker(garden_cfg, mock=False)
        calls = []

        def fake_ce(q, c, top_k=None):
            calls.append(("ce", top_k))
            return c[:top_k]

        def fake_cot(q, c, top_k=None):
            calls.append(("cot", top_k, len(c)))
            return c[:top_k]

        monkeypatch.setattr(rr.ce, "rerank", fake_ce)
        monkeypatch.setattr(rr.cot, "rerank", fake_cot)
        out = rr.rerank("q", _CANDIDATES, top_k=1)
        assert calls[0] == ("ce", 2)          # tier 1 → top_k_rerank_ce
        assert calls[1] == ("cot", 1, 2)      # tier 2 → final, fed tier-1's 2
        assert len(out) == 1

    def test_cot_disabled_uses_ce_order(self, garden_cfg, monkeypatch):
        garden_cfg.reranker.cot_enabled = False
        garden_cfg.retrieval.top_k_rerank_ce = 3
        rr = GardenTwoTierReranker(garden_cfg, mock=False)
        monkeypatch.setattr(rr.ce, "rerank", lambda q, c, top_k=None: c[:top_k])
        out = rr.rerank("q", _CANDIDATES, top_k=2)
        assert [d["chunk_id"] for d in out] == ["a", "b"]

    def test_empty_candidates(self, garden_cfg):
        assert GardenTwoTierReranker(garden_cfg, mock=True).rerank("q", [], top_k=5) == []


class TestConfigFileLoads:
    def test_config_garden_yaml_loads(self):
        cfg = load_config(_CONFIG_GARDEN)
        assert cfg.backend == "garden"
        assert cfg.garden.embed_model == "Qwen3-Embedding-8B"
        assert cfg.garden.llm_model == "google/gemma-3-12b-it"
        assert cfg.garden.embed_region == "asia-northeast1"
        assert cfg.garden.region == "asia-southeast1"
        assert cfg.embedding.dimension == 4096
        assert cfg.qdrant.vector_size == 4096
        assert cfg.qdrant.collection == "legal_vn_garden"
        assert cfg.retrieval.top_k_rerank == 5
        # TIP-013 additions
        assert cfg.retrieval.top_k_fusion == 80
        assert cfg.retrieval.top_k_rerank_ce == 15
        assert cfg.embedding.query_instruction          # non-empty
        assert cfg.reranker.cot_enabled is True
        assert cfg.garden.ce_model == "BAAI/bge-reranker-v2-m3"
        assert cfg.generator.max_articles == 5
