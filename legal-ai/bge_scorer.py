"""
bge_scorer.py — shared BGE-reranker-v2-m3 cross-encoder scoring.

The online answer pipeline class does not invoke BGE directly, but the proven
submission workflow uses this scorer for intent-wise compression. Both Vertex
and GPU configurations therefore share the same BGE endpoint contract.

Endpoint shape — a Vertex AI *dedicated* endpoint serving the model in a TEI
container, called via the :predict API (NOT the OpenAI route), exactly like the
Qwen3 embedder (see GpuEmbedder._embed):

    POST https://{bge_dns}/v1/projects/{proj}/locations/{region}/
         endpoints/{id}:predict
    body  {"instances": [{"inputs": ["<query>", "<doc>"]}, ...]}   # string PAIR
    resp  {"predictions": [[{"label":"LABEL_0","score":<float>}], ...]}

The request contract was verified live: each instance carries `inputs` = a
[query, text] string pair (NOT {"query","text"}); the score is the cross-encoder
relevance (≈1 relevant, →0 irrelevant). `_coerce_score` unwraps the
list-of-one-dict (and tolerates a bare float / {"score":..} for other TEI
versions); any other shape raises so we never silently mis-rank. Auth reuses the
GPU backend's ADC bearer-token helpers (refreshed on 401 via retry_transient).
"""
from __future__ import annotations

import hashlib

from backends_common import retry_transient

# Reuse the GPU backend's cached-ADC token + reset helpers so a single Google
# credential is shared across embedder and BGE scorer (and refreshed on 401).
from gpu_backends import _gcp_token, _reset_creds

# Document text fed to BGE = "{dieu_title} {content}", capped so a pathologically
# long article can't blow up latency / the model's context. BGE-reranker-v2-m3
# handles long inputs, but the discriminative signal is overwhelmingly in the
# head of a legal article; 1000 chars comfortably covers a normal Điều.
_DOC_MAXCHARS = 1000


class BgeError(Exception):
    """Raised when a BGE :predict call fails after retries, or returns garbage."""


class BgeScorer:
    """
    Score query–document pairs with BGE-reranker-v2-m3 on a Vertex GPU endpoint.

    config must expose a `bge` section (BgeConfig): `endpoint_id`, `dns`,
    `region`, `project_id`, `max_texts`, `timeout`, `max_retries`. This endpoint
    is SHARED by both backends (gpu + vertex_ai), so BgeScorer reads `config.bge`
    rather than any backend-specific section.

    dry_run=True returns deterministic pseudo-scores (no network) so the A/B
    plumbing can be exercised offline before the endpoint is live.
    """

    def __init__(self, config, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._b = config.bge

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """
        Score `query` against each document. Returns one float per document,
        SAME order as `documents` (higher = more relevant). Batches by
        gpu.bge_max_texts (one :predict call per batch).
        """
        if not documents:
            return []
        if self.dry_run:
            return [self._mock_score(query, d) for d in documents]

        cap = self._b.max_texts or 32
        scores: list[float] = []
        for start in range(0, len(documents), cap):
            scores.extend(self._score_batch(query, documents[start:start + cap]))
        return scores

    def ping(self, timeout: float = 12.0) -> tuple[bool, str]:
        """
        Single, NON-retried :predict probe to fail fast on a misconfigured /
        un-deployed endpoint (a wrong DNS otherwise hangs for minutes once
        retry_transient multiplies it across batches × questions). Returns
        (ok, human-readable detail). Never raises.
        """
        if self.dry_run:
            return True, "dry_run (mock scorer, no endpoint)"
        b = self._b
        if not b.endpoint_id:
            return False, "bge.endpoint_id empty — set GPU_BGE_ENDPOINT_ID in .env"
        host = b.dns or f"{b.region}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/projects/{b.project_id}/"
            f"locations/{b.region}/endpoints/{b.endpoint_id}:predict"
        )
        try:
            import requests  # lazy
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {_gcp_token()}"},
                json={"instances": [{"inputs": ["ping", "ping"]}]},
                timeout=timeout,
            )
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}: {r.text[:150]}"
            preds = r.json().get("predictions")
            if not preds:
                return False, f"HTTP 200 but no predictions: {r.text[:150]}"
            return True, f"OK — sample score {self._coerce_score(preds[0]):.3f}"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:200]}"

    def score_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Score a list of candidate article dicts (need 'dieu_title' + 'content').
        Adds a 'bge_score' field to each candidate IN PLACE and returns a NEW list
        sorted by bge_score descending (the input list's order is left untouched,
        so callers can keep the original RRF order alongside the BGE order).
        """
        docs = [self._doc_text(c) for c in candidates]
        scores = self.score_pairs(query, docs)
        for c, s in zip(candidates, scores):
            c["bge_score"] = s
        return sorted(candidates, key=lambda c: c.get("bge_score", float("-inf")),
                      reverse=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_text(c: dict) -> str:
        title = (c.get("dieu_title") or "").strip()
        content = (c.get("content") or "").strip()[:_DOC_MAXCHARS]
        return f"{title} {content}".strip()

    def _score_batch(self, query: str, batch: list[str]) -> list[float]:
        b = self._b
        # Dedicated endpoint → its own domain; otherwise the shared regional one.
        host = b.dns or f"{b.region}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/projects/{b.project_id}/"
            f"locations/{b.region}/endpoints/{b.endpoint_id}:predict"
        )
        # TEI cross-encoder predict contract (verified live): each instance is
        # {"inputs": [query, text]} — a string pair, NOT {"query","text"}.
        body = {"instances": [{"inputs": [query, d]} for d in batch]}

        def call() -> list[float]:
            import requests  # lazy
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {_gcp_token()}"},
                json=body,
                timeout=b.timeout,
            )
            if resp.status_code != 200:
                # Surface the status so retry_transient can classify 401/429/5xx.
                raise BgeError(f"predict {resp.status_code}: {resp.text[:200]}")
            preds = resp.json().get("predictions")
            if preds is None:
                raise BgeError("predict returned no 'predictions'")
            if len(preds) != len(batch):
                raise BgeError(
                    f"predict returned {len(preds)} scores for {len(batch)} docs"
                )
            return [self._coerce_score(p) for p in preds]

        try:
            return retry_transient(
                call, attempts=b.max_retries + 2, base=4.0, refresh_auth=_reset_creds
            )
        except BgeError:
            raise
        except Exception as e:
            raise BgeError(f"bge score failed: {e}") from e

    @staticmethod
    def _coerce_score(p) -> float:
        """TEI rerank prediction → float (bare number / {'score':..} / [num])."""
        if isinstance(p, bool):   # guard: bool is an int subclass
            return float(p)
        if isinstance(p, (int, float)):
            return float(p)
        if isinstance(p, dict):
            for k in ("score", "relevance", "logit", "value"):
                if k in p:
                    return float(p[k])
        if isinstance(p, list) and p:
            return BgeScorer._coerce_score(p[0])
        raise BgeError(f"cannot parse BGE score from prediction: {p!r}")

    @staticmethod
    def _mock_score(query: str, doc: str) -> float:
        """Deterministic pseudo-score in [0,1) for dry-run plumbing tests."""
        h = hashlib.sha1((query[:40] + "||" + doc[:120]).encode("utf-8")).hexdigest()
        return int(h[:6], 16) / float(0xFFFFFF)
