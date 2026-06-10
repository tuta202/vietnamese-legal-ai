"""Tests for config validation and ${ENV_VAR} secret expansion."""
from __future__ import annotations

from retrieval.config import (
    RetrievalConfig,
    _expand_env,
    load_config,
    validate_config,
)


class TestValidateConfig:
    def test_vertex_missing_all_creds(self):
        errs = validate_config(RetrievalConfig(backend="vertex_ai"))
        assert any("qdrant.url" in e for e in errs)
        assert any("qdrant.api_key" in e for e in errs)
        assert any("GCP" in e or "GOOGLE_API_KEY" in e for e in errs)

    def test_vertex_ok_with_creds(self):
        cfg = RetrievalConfig(backend="vertex_ai")
        cfg.qdrant.url = "https://x.qdrant.io"
        cfg.qdrant.api_key = "secret"
        cfg.vllm.gcp_project = "proj"
        assert validate_config(cfg) == []

    def test_gpu_ok_with_endpoints(self):
        cfg = RetrievalConfig(backend="gpu")
        cfg.gpu.embed_endpoint_id = "e1"
        cfg.gpu.embed_dns = "embed.example.goog"
        cfg.gpu.llm_endpoint_id = "l1"
        cfg.gpu.project_id = "proj"
        assert validate_config(cfg) == []

    def test_gpu_missing_endpoints_flagged(self):
        errs = validate_config(RetrievalConfig(backend="gpu"))
        assert any("embed_endpoint_id" in e for e in errs)

    def test_unknown_backend_flagged(self):
        errs = validate_config(RetrievalConfig(backend="nope"))
        assert any("unknown backend" in e for e in errs)

    def test_vllm_backend_now_unknown(self):
        # The local vLLM backend was removed; "vllm" is no longer a valid backend.
        errs = validate_config(RetrievalConfig(backend="vllm"))
        assert any("unknown backend" in e for e in errs)


class TestEnvExpansion:
    def test_expand_scalar_and_nested(self, monkeypatch):
        monkeypatch.setenv("LEGALAI_TEST_VAR", "resolved")
        assert _expand_env("${LEGALAI_TEST_VAR}") == "resolved"
        assert _expand_env({"a": ["${LEGALAI_TEST_VAR}"]}) == {"a": ["resolved"]}

    def test_missing_var_expands_empty(self, monkeypatch):
        monkeypatch.delenv("LEGALAI_NOPE", raising=False)
        assert _expand_env("${LEGALAI_NOPE}") == ""

    def test_load_config_resolves_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LEGALAI_TEST_QURL", "https://cluster.qdrant.io")
        p = tmp_path / "c.yaml"
        p.write_text(
            "backend: vertex_ai\nqdrant:\n  url: ${LEGALAI_TEST_QURL}\n",
            encoding="utf-8",
        )
        cfg = load_config(p)
        assert cfg.qdrant.url == "https://cluster.qdrant.io"
