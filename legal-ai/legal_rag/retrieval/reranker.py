"""
LegalReranker — cross-encoder reranker using Qwen3-Reranker.

Shared base class — the gpu/vertex backends override `_score` with a whole-pool
LLM 0–10 scorer; the base cross-encoder is a local fallback.

mock=True: returns candidates[:top_k] in original order; no model loaded.
Model is loaded lazily on first call to rerank().
"""
from __future__ import annotations

from legal_rag.retrieval.config import RetrievalConfig

_INSTRUCTION = (
    "Cho câu hỏi pháp lý sau, hãy đánh giá mức độ liên quan của điều luật đến câu hỏi. "
    "Trả lời 'yes' nếu điều luật liên quan trực tiếp, 'no' nếu không liên quan."
)


class LegalReranker:
    """
    Cross-encoder reranker for retrieved legal articles.
    Importing this module never triggers a model load.
    """

    def __init__(self, config: RetrievalConfig, mock: bool = False) -> None:
        self.config = config
        self.mock   = mock
        self._model = None   # lazy

    # ------------------------------------------------------------------
    # Lazy model property
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is None and not self.mock:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(
            self.config.models.reranker,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidate articles for a question.
        Returns at most top_k articles (or config.retrieval.top_k_rerank).
        In mock mode: preserves original order, just truncates to top_k.
        """
        k = top_k if top_k is not None else self.config.retrieval.top_k_rerank
        if not candidates:
            return []

        if self.mock:
            return candidates[:k]

        scores = self._score(question, candidates)
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in ranked[:k]]

    def rerank_with_scores(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> tuple[list[dict], list[dict]]:
        """
        Rerank AND return the full scored pool for offline threshold sweeping.

        Shared by every backend (uses the overridden `_score`), so score-saving
        works identically on the gpu and vertex backends. Returns
        ``(top_k_results, full_scored_pool)`` where each pool entry is
        ``{relevant_article_str, relevant_doc_str, llm_score, rank_in_pool, in_final}``
        sorted by score descending.
        """
        k = top_k
        if not candidates:
            return [], []

        if self.mock:
            pool = [
                {
                    "relevant_article_str": c.get("relevant_article_str", ""),
                    "relevant_doc_str": c.get("relevant_doc_str", ""),
                    "llm_score": 0.0,
                    "rank_in_pool": i,
                    "in_final": i < k,
                }
                for i, c in enumerate(candidates)
            ]
            return candidates[:k], pool

        scores = self._score(query, candidates)
        pool = [
            {
                "relevant_article_str": candidates[i].get("relevant_article_str", ""),
                "relevant_doc_str": candidates[i].get("relevant_doc_str", ""),
                "llm_score": float(scores[i]),
                "rank_in_pool": i,
                "in_final": False,
            }
            for i in range(len(candidates))
        ]
        sorted_idx = sorted(range(len(pool)), key=lambda i: pool[i]["llm_score"], reverse=True)
        for i in sorted_idx[:k]:
            pool[i]["in_final"] = True
        top_k_results = [candidates[i] for i in sorted_idx[:k]]
        pool.sort(key=lambda x: x["llm_score"], reverse=True)
        return top_k_results, pool

    # ------------------------------------------------------------------
    # Scoring hook — overridden by gpu/vertex subclasses (only this differs)
    # ------------------------------------------------------------------

    def _score(self, question: str, candidates: list[dict]) -> list[float]:
        """Relevance score per candidate (cross-encoder), aligned to input order."""
        pairs = [
            (
                f"{_INSTRUCTION}\nQuestion: {question}",
                f"{c.get('dieu_number', '')} {c.get('dieu_title', '')}\n{c.get('content', '')[:512]}",
            )
            for c in candidates
        ]
        return self.model.predict(pairs)
