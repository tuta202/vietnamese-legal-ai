"""
RetrievalConfig — config-driven, all modules receive config via constructor.
Loads from YAML; falls back to defaults if file not found.

Unified schema: the same dataclasses serve both backends. `backend` selects
between the vLLM/Qwen stack ("vllm") and the Vertex AI/Gemini stack
("vertex_ai"); only the model-call layer differs (see vertex_backends.py).

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

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
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
    """Chat-LLM endpoint config (used by rewriter, generator, LLM reranker)."""
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    # Vertex AI fields — empty/default for the local vLLM backend:
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


@dataclass
class GeneratorConfig:
    temperature: float = 0.3
    max_tokens: int = 3000
    top_p: float = 0.9
    max_articles: int = 7   # max context articles passed to the LLM


@dataclass
class RetrievalConfig:
    backend: str = "vllm"   # "vllm" | "vertex_ai"
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


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
        if not cfg.qdrant.api_key:
            errors.append("qdrant.api_key is empty — set QDRANT_API_KEY in .env")
        if not cfg.vllm.gcp_project and not os.environ.get("GOOGLE_API_KEY"):
            errors.append(
                "no GCP auth — set GCP_PROJECT (Vertex/ADC) or GOOGLE_API_KEY in .env"
            )
    elif cfg.backend == "vllm":
        if not cfg.vllm.base_url:
            errors.append("vllm.base_url is empty")
        if not cfg.qdrant.host:
            errors.append("qdrant.host is empty")
    else:
        errors.append(f"unknown backend '{cfg.backend}' (expected 'vllm' or 'vertex_ai')")
    return errors
