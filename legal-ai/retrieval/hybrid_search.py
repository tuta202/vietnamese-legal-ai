"""
HybridSearcher — combines BM25 + dense retrieval with RRF fusion.

All dependencies injected via constructor; no hardcoded singletons.
Designed so every dependency can be a mock for offline testing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from retrieval.bm25_index import BM25Index
from retrieval.config import RetrievalConfig
from retrieval.embedder import LegalEmbedder
from retrieval.query_rewriter import QueryRewriter


# ---------------------------------------------------------------------------
# RRF helper (standalone — importable for unit tests)
# ---------------------------------------------------------------------------

def rrf_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        rankings: List of [(chunk_id, score), ...] sorted descending by score.
        k: RRF constant (default 60).

    Returns:
        Sorted list of (chunk_id, rrf_score) descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# HybridSearcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Full hybrid retrieval pipeline:
      1. Query rewriting (rewritten_query + topic_description)
      2. BM25 search on rewritten query
      3. Dense search on rewritten query vector
      4. Dense search on topic_description vector (if available)
      5. RRF fusion of all result lists

    qdrant_client: any object with .search(collection_name, query_vector, limit).
    Returns list of payload dicts with extra field "rrf_score".
    """

    def __init__(
        self,
        config: RetrievalConfig,
        bm25: BM25Index,
        embedder: LegalEmbedder,
        rewriter: QueryRewriter,
        qdrant_client: Any,
    ) -> None:
        self.config  = config
        self._bm25   = bm25
        self._embed  = embedder
        self._rewrite = rewriter
        self._qdrant = qdrant_client

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Run hybrid retrieval for a question.
        Returns top_k (or config.retrieval.top_k_fusion) dicts with 'rrf_score'.
        """
        k_fusion = top_k if top_k is not None else self.config.retrieval.top_k_fusion

        # Step 1 — Query rewriting
        rewrite          = self._rewriter_call(question)
        rewritten        = rewrite["rewritten_query"]
        topic_description = rewrite["topic_description"]

        # Step 2 — BM25
        bm25_hits = self._bm25.search(rewritten, top_k=self.config.retrieval.top_k_bm25)

        # Step 3 — Dense: rewritten query
        q_vec      = self._embed.embed_query(rewritten)
        dense_hits = self._dense_search(q_vec)

        # Step 4 — Dense: topic description (only if non-trivial)
        rankings = [bm25_hits, dense_hits]
        if topic_description and topic_description != rewritten:
            topic_vec  = self._embed.embed_query(topic_description)
            topic_hits = self._dense_search(topic_vec)
            rankings.append(topic_hits)

        # Step 5 — RRF fusion
        fused = rrf_fusion(rankings, k=self.config.retrieval.rrf_k)[:k_fusion]

        return self._resolve_payloads(fused)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rewriter_call(self, question: str) -> dict:
        return self._rewrite.rewrite(question)

    def _dense_search(self, vector: np.ndarray) -> list[tuple[str, float]]:
        """Call Qdrant (query_points) and return [(chunk_id, score), ...]."""
        resp = self._qdrant.query_points(
            collection_name=self.config.qdrant.collection,
            query=vector.tolist(),
            limit=self.config.retrieval.top_k_dense,
            with_payload=True,
        )
        results = getattr(resp, "points", resp)
        # Handle both real ScoredPoint and mock dicts
        out = []
        for r in results:
            if hasattr(r, "id") and hasattr(r, "score"):
                out.append((str(r.id), float(r.score)))
            elif isinstance(r, dict):
                out.append((str(r["id"]), float(r.get("score", 0.0))))
        return out

    def _resolve_payloads(self, fused: list[tuple[str, float]]) -> list[dict]:
        """Attach article payload to each (chunk_id, rrf_score) pair."""
        results = []
        for chunk_id, rrf_score in fused:
            payload = self._bm25.get_payload(chunk_id)
            if payload is None:
                # Try Qdrant for dense-only hits
                payload = self._qdrant_payload(chunk_id)
            if payload is None:
                payload = {"chunk_id": chunk_id}
            results.append({**payload, "rrf_score": rrf_score})
        return results

    def _qdrant_payload(self, chunk_id: str) -> dict | None:
        """Retrieve payload from Qdrant for a chunk not in BM25 index."""
        try:
            points = self._qdrant.retrieve(
                collection_name=self.config.qdrant.collection,
                ids=[chunk_id],
                with_payload=True,
            )
            if points:
                p = points[0]
                return p.payload if hasattr(p, "payload") else None
        except Exception:
            return None
        return None
