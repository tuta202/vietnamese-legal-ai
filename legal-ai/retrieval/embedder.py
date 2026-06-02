"""
LegalEmbedder — lazy-loading dense embedder for Qwen3-Embedding.

dry_run=True: returns deterministic zero-vectors; no model loaded.
Model is loaded on first call to embed_query() / embed_corpus().
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from retrieval.config import RetrievalConfig, load_config

if TYPE_CHECKING:
    pass  # qdrant_client imported lazily below

_EMBED_DIM = 4096   # Qwen3-Embedding-8B output dimension
_EMBED_INSTRUCTION = (
    "Tìm kiếm điều luật pháp lý Việt Nam liên quan đến câu hỏi sau"
)
_BATCH_SIZE = 32


class LegalEmbedder:
    """
    Dense embedder wrapping Qwen3-Embedding.
    Heavy model is loaded lazily; importing this module never triggers a load.
    """

    def __init__(self, config: RetrievalConfig, dry_run: bool = False) -> None:
        self.config   = config
        self.dry_run  = dry_run
        self._model   = None   # loaded on first use

    # ------------------------------------------------------------------
    # Lazy model property
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is None and not self.dry_run:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            self.config.models.embedder,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # Text formatting
    # ------------------------------------------------------------------

    def _format_query(self, text: str) -> str:
        return f"Instruct: {_EMBED_INSTRUCTION}\nQuery: {text}"

    def _format_document(self, article: dict) -> str:
        header = (
            f"{article.get('law_id', '')} | "
            f"{article.get('law_type', '')} {article.get('law_name', '')} | "
            f"{article.get('dieu_number', '')}"
        )
        title   = article.get("dieu_title", "")
        content = article.get("content", "")[:512]
        return f"{header}\n{title}\n{content}"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32 array."""
        if self.dry_run:
            return np.zeros(_EMBED_DIM, dtype=np.float32)
        formatted = self._format_query(text)
        vec = self.model.encode(formatted, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of pre-formatted document texts."""
        if self.dry_run:
            return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=_BATCH_SIZE)
        return np.array(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Corpus upsert
    # ------------------------------------------------------------------

    def embed_corpus(
        self,
        articles: list[dict],
        qdrant_client=None,
    ) -> int:
        """
        Embed all articles and upsert to Qdrant.
        In dry_run mode: skips qdrant upsert; returns article count.
        Returns: number of points upserted.
        """
        if self.dry_run:
            print(f"  [dry_run] Would embed and upsert {len(articles)} articles")
            return len(articles)

        try:
            from qdrant_client.models import PointStruct
        except ImportError as e:
            raise RuntimeError("qdrant_client not installed") from e

        upserted = 0
        for i in range(0, len(articles), _BATCH_SIZE):
            batch = articles[i : i + _BATCH_SIZE]
            texts = [self._format_document(a) for a in batch]
            vectors = self.embed_documents(texts)

            points = []
            for art, vec in zip(batch, vectors):
                point_id = str(uuid.UUID(art["chunk_id"]))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload={
                            "chunk_id":             art["chunk_id"],
                            "law_id":               art["law_id"],
                            "law_type":             art["law_type"],
                            "law_name":             art["law_name"],
                            "dieu_number":          art["dieu_number"],
                            "dieu_title":           art["dieu_title"],
                            "content":              art["content"],
                            "relevant_doc_str":     art["relevant_doc_str"],
                            "relevant_article_str": art["relevant_article_str"],
                        },
                    )
                )

            qdrant_client.upsert(
                collection_name=self.config.qdrant.collection,
                points=points,
                wait=True,
            )
            upserted += len(points)
            print(f"  Upserted {upserted}/{len(articles)}", end="\r", flush=True)

        print()
        return upserted
