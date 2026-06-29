"""
RetrievalConfig — config-driven, all modules receive config via constructor.
Loads from YAML; falls back to defaults if file not found.

Unified schema: the same dataclasses serve both backends. `backend` selects
between the self-deployed open-source stack ("gpu": Qwen3 + Gemma on GPU
endpoints) and the Vertex AI/Gemini stack ("vertex_ai"); only the model-call
layer differs (see gpu_backends.py / vertex_backends.py).

Secrets are never stored in the YAML — values like `${QDRANT_API_KEY}` are
expanded from the process environment or a gitignored `.env` next to the
repo root at load time.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_gpu_clean.yaml"
# .env lives at the project root (legal-ai/), one level up from retrieval/.
_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection: str = "legal_vn"
    # Cloud (Vertex) fields — empty for the local vLLM backend:
    url: str = ""
    api_key: str = ""
    vector_size: int = 4096
    distance: str = "Cosine"


@dataclass
class VllmConfig:
    """
    Chat-LLM endpoint config (used by rewriter, generator, LLM reranker).

    Kept under the `vllm` key for schema parity with the shared base classes.
    The vertex_ai backend uses `model` + `gcp_project`/`gcp_location` (Gemini);
    the gpu backend ignores this section entirely and uses `gpu.*`.
    `base_url` is only a fallback for the base components and is unused by either
    real backend (both override the model-call hook).
    """
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    gcp_project: str = ""
    gcp_location: str = "us-central1"


@dataclass
class ModelsConfig:
    embedder: str = "Qwen/Qwen3-Embedding-8B"
    reranker: str = "Qwen/Qwen3-Reranker-4B"


@dataclass
class EmbeddingConfig:
    dimension: int = 4096
    task_type: str = "RETRIEVAL_DOCUMENT"
    query_task_type: str = "RETRIEVAL_QUERY"
    batch_size: int = 32


@dataclass
class RerankerConfig:
    temperature: float = 0.0


@dataclass
class RetrievalParams:
    # Candidate-pool defaults sized for the expanded ~113k-article corpus (was
    # 20/20/20 when the corpus held 1044 articles). A wider net protects recall
    # against the ~100x larger set of competing documents; the reranker still
    # narrows the pool down to top_k_rerank for the final citations.
    top_k_dense: int = 50
    top_k_bm25: int = 50
    top_k_fusion: int = 40
    top_k_rerank: int = 7
    rrf_k: int = 60
    enable_intent_retrieval: bool = True
    intent_top_k_bm25: int = 50
    intent_top_k_dense: int = 50
    intent_top_k_rrf: int = 10


@dataclass
class GeneratorConfig:
    temperature: float = 0.3
    max_tokens: int = 3000
    top_p: float = 0.9
    max_articles: int = 7   # max context articles passed to the LLM


@dataclass
class GpuConfig:
    """
    Self-deployed GPU backend (open-source, competition-compliant): all components
    run open-source models on Vertex AI Online Endpoints backed by GPU (no Gemini
    fallback; use backend=vertex_ai for that).

    Flat by design so the existing `_section` loader parses it without nesting.
    Endpoints are OpenAI-compatible vLLM serving; auth is a refreshed google-auth
    bearer token (see gpu_backends.py).
    """
    # Embedder endpoint (Qwen3-Embedding-8B). Deployed as a Vertex dedicated
    # endpoint (TEI predict API, NOT OpenAI), so it is called via its dedicated
    # domain `embed_dns` with a :predict request — see GpuEmbedder._embed.
    embed_endpoint_id: str = ""
    embed_model: str = "Qwen3-Embedding-8B"
    embed_dns: str = ""   # dedicated domain, e.g. <id>.<region>-<proj#>.prediction.vertexai.goog
    embed_region: str = ""   # region of the embed endpoint; falls back to `region` if empty
    # LLM + reranker endpoint (Gemma 3 12B-it — same endpoint, two roles).
    llm_endpoint_id: str = ""
    llm_model: str = "google/gemma-3-12b-it"
    llm_dns: str = ""   # dedicated domain for the LLM endpoint; empty → shared aiplatform domain
    # GCP project + the LLM endpoint's region (embed may live in another region;
    # see embed_region). The two endpoints can be in DIFFERENT regions.
    project_id: str = ""
    region: str = "us-central1"
    max_retries: int = 3
    timeout: int = 120


@dataclass
class BgeConfig:
    """
    BGE reranker endpoint (BAAI/bge-reranker-v2-m3) — a cross-encoder deployed on
    GPU (Vertex dedicated endpoint, TEI :predict route). SHARED by both backends:
    the gpu and vertex_ai stacks both call this same endpoint for reranking. Used
    by the intent-wise compression stage through bge_scorer.py. `dns` empty →
    the shared regional aiplatform domain is used
    (only works for non-dedicated endpoints); set it for a dedicated endpoint.
    """
    endpoint_id: str = ""
    dns: str = ""
    region: str = "asia-northeast1"
    project_id: str = ""
    max_texts: int = 32   # hard cap on docs per :predict call
    timeout: int = 120
    max_retries: int = 3


@dataclass
class RetrievalConfig:
    backend: str = "gpu"   # "gpu" | "vertex_ai"
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    bge: BgeConfig = field(default_factory=BgeConfig)


# ---------------------------------------------------------------------------
# Secret handling — .env + ${ENV_VAR} expansion (no third-party dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path | str = _DOTENV_PATH) -> None:
    """Load KEY=value lines from .env. Real env vars take precedence."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${ENV_VAR} references inside strings."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    return obj


def load_config(path: Path | str = _DEFAULT_CONFIG_PATH) -> RetrievalConfig:
    """Load RetrievalConfig from a YAML file; uses defaults if file is missing."""
    path = Path(path)
    if not path.exists():
        return RetrievalConfig()

    try:
        import yaml  # PyYAML
    except ImportError:
        return RetrievalConfig()

    _load_dotenv()
    raw: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _expand_env(raw)

    cfg = RetrievalConfig()
    if "backend" in raw:
        cfg.backend = str(raw["backend"])

    def _section(name: str, cls):
        data = raw.get(name)
        if isinstance(data, dict):
            return cls(**{k: v for k, v in data.items()
                          if k in cls.__dataclass_fields__})
        return None

    for name, cls, attr in [
        ("qdrant",    QdrantConfig,    "qdrant"),
        ("vllm",      VllmConfig,      "vllm"),
        ("models",    ModelsConfig,    "models"),
        ("embedding", EmbeddingConfig, "embedding"),
        ("reranker",  RerankerConfig,  "reranker"),
        ("retrieval", RetrievalParams, "retrieval"),
        ("generator", GeneratorConfig, "generator"),
        ("gpu",    GpuConfig,    "gpu"),
        ("bge",    BgeConfig,    "bge"),
    ]:
        parsed = _section(name, cls)
        if parsed is not None:
            setattr(cfg, attr, parsed)

    return cfg


def validate_config(cfg: RetrievalConfig) -> list[str]:
    """
    Return a list of human-readable problems with a config for a real (non-mock)
    run; empty list means OK. Used for fail-fast before any API call.
    """
    errors: list[str] = []
    if cfg.backend == "vertex_ai":
        if not cfg.qdrant.url:
            errors.append("qdrant.url is empty — set QDRANT_URL in .env")
        # A local Qdrant (localhost) needs no api_key; only require it for remote.
        is_local = any(h in cfg.qdrant.url for h in ("localhost", "127.0.0.1"))
        if not is_local and not cfg.qdrant.api_key:
            errors.append("qdrant.api_key is empty — set QDRANT_API_KEY in .env")
        if not cfg.vllm.gcp_project and not os.environ.get("GOOGLE_API_KEY"):
            errors.append(
                "no GCP auth — set GCP_PROJECT (Vertex/ADC) or GOOGLE_API_KEY in .env"
            )
    elif cfg.backend == "gpu":
        _validate_gpu(cfg, errors)
    else:
        errors.append(
            f"unknown backend '{cfg.backend}' "
            "(expected 'gpu' or 'vertex_ai')"
        )
    return errors


def _validate_gpu(cfg: RetrievalConfig, errors: list[str]) -> None:
    """Validation for the pure self-deployed GPU endpoints backend."""
    g = cfg.gpu

    # Qdrant: a URL means remote (needs api_key unless localhost); otherwise the
    # local host/port pair is used.
    if cfg.qdrant.url:
        is_local = any(h in cfg.qdrant.url for h in ("localhost", "127.0.0.1"))
        if not is_local and not cfg.qdrant.api_key:
            errors.append("qdrant.api_key is empty — set QDRANT_API_KEY in .env")
    elif not cfg.qdrant.host:
        errors.append("qdrant.host is empty")

    # Both endpoints are required — every component runs on the GPU backend.
    if not g.embed_endpoint_id:
        errors.append("gpu.embed_endpoint_id empty — set GPU_EMBED_ENDPOINT_ID in .env")
    if not g.embed_dns:
        errors.append("gpu.embed_dns empty — set GPU_EMBED_DNS in .env (dedicated domain)")
    if not g.llm_endpoint_id:
        errors.append("gpu.llm_endpoint_id empty — set GPU_LLM_ENDPOINT_ID in .env")
    if not g.project_id:
        errors.append("gpu.project_id empty — set GCP_PROJECT in .env")

    # The dense vector size must match the embedder's output dimension, or every
    # Qdrant upsert/search will fail at runtime.
    if cfg.embedding.dimension != cfg.qdrant.vector_size:
        errors.append(
            f"embedding.dimension ({cfg.embedding.dimension}) != "
            f"qdrant.vector_size ({cfg.qdrant.vector_size})"
        )
