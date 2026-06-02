"""
RetrievalConfig — config-driven, all modules receive config via constructor.
Loads from YAML; falls back to defaults if file not found.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection: str = "legal_vn"


@dataclass
class VllmConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class ModelsConfig:
    embedder: str = "Qwen/Qwen3-Embedding-8B"
    reranker: str = "Qwen/Qwen3-Reranker-4B"


@dataclass
class RetrievalParams:
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_fusion: int = 20
    top_k_rerank: int = 7
    rrf_k: int = 60


@dataclass
class GeneratorConfig:
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    max_articles: int = 7   # max context articles passed to the LLM


@dataclass
class RetrievalConfig:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


def _from_dict(cls, data: dict) -> Any:
    """Recursively populate a dataclass from a dict."""
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            continue
        val = data[name]
        # Recurse into nested dataclasses
        if hasattr(f.type, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[name] = _from_dict(f.type, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(path: Path | str = _DEFAULT_CONFIG_PATH) -> RetrievalConfig:
    """Load RetrievalConfig from a YAML file; uses defaults if file is missing."""
    path = Path(path)
    if not path.exists():
        return RetrievalConfig()

    try:
        import yaml  # PyYAML
    except ImportError:
        return RetrievalConfig()

    raw: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cfg = RetrievalConfig()
    if "qdrant" in raw:
        cfg.qdrant = QdrantConfig(**{k: v for k, v in raw["qdrant"].items()
                                     if k in QdrantConfig.__dataclass_fields__})
    if "vllm" in raw:
        cfg.vllm = VllmConfig(**{k: v for k, v in raw["vllm"].items()
                                  if k in VllmConfig.__dataclass_fields__})
    if "models" in raw:
        cfg.models = ModelsConfig(**{k: v for k, v in raw["models"].items()
                                      if k in ModelsConfig.__dataclass_fields__})
    if "retrieval" in raw:
        cfg.retrieval = RetrievalParams(**{k: v for k, v in raw["retrieval"].items()
                                           if k in RetrievalParams.__dataclass_fields__})
    if "generator" in raw:
        cfg.generator = GeneratorConfig(**{k: v for k, v in raw["generator"].items()
                                           if k in GeneratorConfig.__dataclass_fields__})
    return cfg
