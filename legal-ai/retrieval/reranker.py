"""
LegalReranker — cross-encoder reranker using Qwen3-Reranker.

mock=True: returns candidates[:top_k] in original order; no model loaded.
Model is loaded lazily on first call to rerank().
"""
from __future__ import annotations

from retrieval.config import RetrievalConfig

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

        pairs = [
            (
                f"{_INSTRUCTION}\nQuestion: {question}",
                f"{c.get('dieu_number', '')} {c.get('dieu_title', '')}\n{c.get('content', '')[:512]}",
            )
            for c in candidates
        ]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in ranked[:k]]
