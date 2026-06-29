"""
LegalEmbedder — lazy-loading dense embedder for Qwen3-Embedding.

Shared base class — the gpu/vertex backends override the encode hooks to call
their respective endpoints; the base loads a local model only as a fallback.

dry_run=True: returns deterministic zero-vectors; no model loaded.
Model is loaded on first call to embed_query() / embed_corpus().
"""
from __future__ import annotations

import hashlib
import logging
import unicodedata
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from legal_rag.retrieval.config import RetrievalConfig, load_config

if TYPE_CHECKING:
    pass  # qdrant_client imported lazily below

log = logging.getLogger(__name__)

_EMBED_DIM = 4096   # Qwen3-Embedding-8B output dimension
_EMBED_INSTRUCTION = (
    "Tìm kiếm điều luật pháp lý Việt Nam liên quan đến câu hỏi sau"
)
_BATCH_SIZE = 32    # default; overridden by config.embedding.batch_size
_LAW_NAME_MAXCHARS = 220
_TITLE_MAXCHARS = 180
_DOC_CONTENT_MAXCHARS = 1000


def format_document_text(article: dict) -> str:
    """Canonical text embedded for one corpus article."""
    law_name = str(article.get("law_name", ""))[:_LAW_NAME_MAXCHARS]
    title = str(article.get("dieu_title", ""))[:_TITLE_MAXCHARS]
    content = str(article.get("content", ""))[:_DOC_CONTENT_MAXCHARS]
    return (
        f"{article.get('law_type', '')} {law_name}\n"
        f"{article.get('dieu_number', '')}. {title}\n"
        f"{content}"
    )


def embedding_text_sha256(article: dict) -> str:
    """Fingerprint the exact normalized document text sent to an embedder."""
    text = unicodedata.normalize("NFC", format_document_text(article))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        return format_document_text(article)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimension (config-driven; 4096 for Qwen3-Embedding-8B)."""
        return getattr(self.config.embedding, "dimension", _EMBED_DIM)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32 array."""
        if self.dry_run:
            return np.zeros(self.dim, dtype=np.float32)
        return self._encode_query(unicodedata.normalize("NFC", text))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of pre-formatted document texts."""
        if self.dry_run:
            return np.zeros((len(texts), self.dim), dtype=np.float32)
        normalized = [unicodedata.normalize("NFC", text) for text in texts]
        return self._encode_documents(normalized)

    # ------------------------------------------------------------------
    # Encoding hooks — overridden by Vertex subclass (only these differ)
    # ------------------------------------------------------------------

    def _encode_query(self, text: str) -> np.ndarray:
        formatted = self._format_query(text)
        vec = self.model.encode(formatted, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        bs = getattr(self.config.embedding, "batch_size", _BATCH_SIZE)
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=bs)
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
            log.info("[dry_run] would embed and upsert %d articles", len(articles))
            return len(articles)

        try:
            from qdrant_client.models import PointStruct
        except ImportError as e:
            raise RuntimeError("qdrant_client not installed") from e

        batch_size = getattr(self.config.embedding, "batch_size", _BATCH_SIZE)
        upserted = 0
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
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
            log.info("Upserted %d/%d", upserted, len(articles))

        return upserted
