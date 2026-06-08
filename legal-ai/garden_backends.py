"""
garden_backends.py — Vertex AI Model Garden (open-source, competition-compliant)
implementations of the four neural components, as subclasses that override ONLY
the model-call hook — exactly mirroring vertex_backends.py.

This backend is PURE Garden: every component runs an open-source model served on
a Vertex AI Online Endpoint. There is no Gemini fallback here — for the Gemini
stack use `backend: vertex_ai` (config_vertex.yaml) instead. Clean separation:
  • vertex_ai → Gemini everywhere
  • garden    → Garden everywhere

  QueryRewriter   → GardenQueryRewriter   (overrides _chat_complete)
  LegalGenerator  → GardenGenerator       (overrides _chat_complete)
  LegalEmbedder   → GardenEmbedder        (overrides _encode_query/_encode_documents)
  LegalReranker   → GardenReranker        (overrides _score)

Endpoints are OpenAI-compatible vLLM servers:
  • Embedder  : Qwen3-Embedding-8B  → 4096-dim vectors
  • LLM/rerank: Gemma 3 12B-it      → generation + 0–10 relevance scoring

Auth is a google-auth bearer token passed as the OpenAI api_key. The token
expires after ~1h, so a fresh client (with a freshly-minted token) is built for
every call; a 401 forces a credential refresh on the next retry.

Heavy SDKs (openai, google.auth) are imported lazily inside methods, so importing
this module — or constructing any class with mock/dry_run=True — never touches
them. Shared, provider-agnostic helpers come from backends_common.
"""
from __future__ import annotations

import logging

import numpy as np

from backends_common import (
    RERANK_COT_SYSTEM,
    RERANK_SNIPPET,
    RERANK_SYSTEM,
    format_cot_rerank_user,
    parse_rerank_scores,
    retry_transient,
)
from generator.llm_generator import LegalGenerator
from retrieval.config import GardenConfig, RetrievalConfig
from retrieval.embedder import LegalEmbedder
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import LegalReranker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GardenError(Exception):
    """Base error for the Model Garden backend."""


class EmbeddingError(GardenError):
    """Raised when an embedding call fails after retries."""


class GenerationError(GardenError):
    """Raised when a chat-completion call fails after retries."""


class RerankError(GardenError):
    """Raised when a rerank scoring call fails after retries."""


# ---------------------------------------------------------------------------
# Auth + client + retry helpers
# ---------------------------------------------------------------------------

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_creds = None   # module-level cached ADC credentials


def _gcp_token() -> str:
    """Return a valid GCP bearer token, refreshing application-default creds."""
    global _creds
    import google.auth  # lazy
    from google.auth.transport.requests import Request  # lazy

    if _creds is None:
        _creds, _ = google.auth.default(scopes=_SCOPES)
    if not _creds.valid:
        _creds.refresh(Request())
    return _creds.token


def _reset_creds() -> None:
    """Drop the cached credentials so the next token() mints a fresh one."""
    global _creds
    _creds = None


def _garden_client(g: GardenConfig, endpoint_id: str):
    """
    Build an OpenAI client for the Garden LLM endpoint. A *dedicated* endpoint
    (llm_dns set) is only reachable on its own domain; a *standard* endpoint uses
    the shared aiplatform domain. Both expose OpenAI /chat/completions at
    .../endpoints/{id}.
    """
    from openai import OpenAI  # lazy

    host = g.llm_dns or f"{g.region}-aiplatform.googleapis.com"
    base_url = (
        f"https://{host}/v1beta1/"
        f"projects/{g.project_id}/locations/{g.region}/endpoints/{endpoint_id}"
    )
    return OpenAI(base_url=base_url, api_key=_gcp_token(), timeout=g.timeout)


def _garden_retry(fn, attempts: int):
    """Garden retry: transient backoff + bearer-token refresh on 401."""
    return retry_transient(fn, attempts=attempts, base=4.0, refresh_auth=_reset_creds)


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

class GardenQueryRewriter(QueryRewriter):
    """Gemma-backed query rewriter; inherits prompts, few-shots, JSON parsing."""

    def _chat_complete(self, messages: list[dict]) -> str:
        g = self.config.garden

        def call() -> str:
            client = _garden_client(g, g.llm_endpoint_id)
            resp = client.chat.completions.create(
                model=g.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""

        try:
            return _garden_retry(call, g.max_retries + 2)
        except Exception as e:
            raise GenerationError(f"garden rewrite failed: {e}") from e


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

class GardenGenerator(LegalGenerator):
    """Gemma-backed generator; inherits the PromptBuilder rules + mock answer."""

    def _chat_complete(self, system: str, user: str) -> str:
        g = self.config.garden
        gen = self.config.generator

        def call() -> str:
            client = _garden_client(g, g.llm_endpoint_id)
            resp = client.chat.completions.create(
                model=g.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                top_p=gen.top_p,
            )
            return resp.choices[0].message.content or ""

        try:
            return _garden_retry(call, g.max_retries + 2)
        except Exception as e:
            raise GenerationError(f"garden generate failed: {e}") from e


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class GardenEmbedder(LegalEmbedder):
    """
    Qwen3-Embedding-8B backed embedder (OpenAI-compatible Garden endpoint).
    Inherits embed_query/embed_documents/embed_corpus; overrides only the encode
    hooks. `dim` comes from config.embedding.dimension (4096 for Qwen3). Queries
    keep the base "Instruct:" wrapper; documents are pre-formatted by the caller
    (embed_corpus / reembed) and embedded as-is, matching the local Qwen path.
    """

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed via the Vertex AI dedicated-endpoint :predict API (TEI container),
        NOT the OpenAI route — this Qwen3 endpoint is a dedicated endpoint and is
        only reachable on its dedicated domain. Request/response shape (verified):
            POST https://{embed_dns}/v1/projects/{proj}/locations/{region}/
                 endpoints/{id}:predict
            body  {"instances": [{"inputs": "<text>"}, ...]}
            resp  {"predictions": [[[...4096 floats...]], ...]}  # vector = pred[0]
        """
        g = self.config.garden
        region = g.embed_region or g.region   # embed may live in another region
        url = (
            f"https://{g.embed_dns}/v1/projects/{g.project_id}/"
            f"locations/{region}/endpoints/{g.embed_endpoint_id}:predict"
        )
        body = {"instances": [{"inputs": t} for t in texts]}

        def call() -> list[list[float]]:
            import requests  # lazy
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {_gcp_token()}"},
                json=body,
                timeout=g.timeout,
            )
            if resp.status_code != 200:
                # Surface the status code so _garden_retry can classify 401/429/5xx.
                raise EmbeddingError(f"predict {resp.status_code}: {resp.text[:200]}")
            preds = resp.json().get("predictions")
            if not preds:
                raise EmbeddingError("predict returned no predictions")
            # Each prediction is [[...vector...]]; unwrap the inner vector.
            return [p[0] if (p and isinstance(p[0], list)) else p for p in preds]

        try:
            return _garden_retry(call, g.max_retries + 2)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"garden embed failed: {e}") from e

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _encode_query(self, text: str) -> np.ndarray:
        formatted = self._format_query(text)
        vec = np.array(self._embed([formatted])[0], dtype=np.float32)
        return self._l2_normalize(vec)

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(self._embed(list(texts)), dtype=np.float32)
        return self._l2_normalize(vecs)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

# Score the WHOLE candidate pool in ONE call (set high enough to cover
# top_k_fusion). Batching into small chunks would score each chunk on an
# independent 0–10 scale with no cross-chunk calibration — a "7" in one call is
# not comparable to a "7" in another. A single call gives the model every
# candidate at once, so the scores share one global scale (this matches the
# Vertex/Gemini reranker, which also scores the whole pool in one call). At
# top_k_fusion=40 the prompt is ~7K tokens — trivial for Gemma 3 12B on A100.
_RERANK_BATCH = 100


class GardenReranker(LegalReranker):
    """
    LLM-based reranker (Gemma). Overrides only the scoring hook; rerank() (mock
    short-circuit + sort + truncate) is inherited unchanged. Scores the whole
    candidate pool in ONE call (_RERANK_BATCH) so all scores share a single,
    globally-calibrated 0–10 scale — see the _RERANK_BATCH note.
    """

    def _score(self, question: str, candidates: list[dict]) -> list[float]:
        scores_map: dict[int, float] = {}
        any_ok = False
        for start in range(0, len(candidates), _RERANK_BATCH):
            chunk = candidates[start:start + _RERANK_BATCH]
            partial = self._llm_scores(question, chunk)
            if partial:
                any_ok = True
                for local_i, sc in partial.items():
                    if 0 <= local_i < len(chunk):
                        scores_map[start + local_i] = sc
        if not any_ok:
            # Total parse/network failure → preserve retrieval order.
            n = len(candidates)
            return [float(n - i) for i in range(n)]
        return [float(scores_map.get(i, 0.0)) for i in range(len(candidates))]

    def _llm_scores(self, question: str, candidates: list[dict]) -> dict[int, float] | None:
        g = self.config.garden
        blocks = [f"Câu hỏi: {question}", "", "Danh sách điều luật cần chấm điểm:"]
        for i, c in enumerate(candidates):
            blocks.append(
                f"[{i}] {c.get('dieu_number', '')} {c.get('dieu_title', '')}\n"
                f"{c.get('content', '')[:RERANK_SNIPPET]}"
            )

        def call() -> str:
            client = _garden_client(g, g.llm_endpoint_id)
            resp = client.chat.completions.create(
                model=g.llm_model,
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM},
                    {"role": "user", "content": "\n".join(blocks)},
                ],
                temperature=self.config.reranker.temperature,
                # Bound the JSON array. Whole-pool scoring (~40 items) needs more
                # room than a 10-item batch did (~40×{"index":N,"score":N}).
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""

        try:
            raw = _garden_retry(call, g.max_retries + 2)
        except Exception:
            return None
        return parse_rerank_scores(raw)


# ---------------------------------------------------------------------------
# TIER 1 — BGE cross-encoder reranker (TEI /rerank)
# ---------------------------------------------------------------------------

_CE_SNIPPET = 1000   # chars of article content sent to the cross-encoder per item


class GardenCrossEncoderReranker(LegalReranker):
    """
    TIER 1 — BAAI/bge-reranker-v2-m3 cross-encoder on a Garden TEI dedicated
    endpoint (:predict route). Each (query, doc) pair is scored INDEPENDENTLY (no context dilution
    or position bias), so it handles the full fused pool well. Uses the ORIGINAL
    query. On endpoint failure it raises RerankError — NO silent fallback: the
    cross-encoder is the main precision gate, so a failure must surface.
    """

    def rerank(self, question: str, candidates: list[dict],
               top_k: int | None = None) -> list[dict]:
        k = top_k if top_k is not None else self.config.retrieval.top_k_rerank_ce
        if not candidates:
            return []
        if self.mock:
            return candidates[:k]
        scores = self._ce_scores(question, candidates)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "ce_score": float(s)} for s, c in ranked[:k]]

    @staticmethod
    def _doc_text(c: dict) -> str:
        return (
            f"{c.get('dieu_number', '')} {c.get('dieu_title', '')}\n"
            f"{c.get('content', '')[:_CE_SNIPPET]}"
        )

    def _ce_scores(self, question: str, candidates: list[dict]) -> list[float]:
        """
        Score via the Vertex dedicated-endpoint :predict route (TEI rerank), NOT a
        native /rerank path — verified against the live endpoint: /rerank returns
        404/UNIMPLEMENTED, while :predict with ONE instance of {query, texts}
        works. Shape (verified):
            POST https://{ce_dns}/v1/projects/{proj}/locations/{region}/
                 endpoints/{id}:predict
            body  {"instances": [{"query": "<q>", "texts": ["<d0>", "<d1>", ...]}]}
            resp  {"predictions": [[{"index": i, "score": s}, ...]]}  # sorted desc
        """
        g = self.config.garden
        region = g.ce_region or g.region   # rerank may live in another region
        url = (
            f"https://{g.ce_dns}/v1/projects/{g.project_id}/"
            f"locations/{region}/endpoints/{g.ce_endpoint_id}:predict"
        )
        body = {
            "instances": [{
                "query": question,                                 # ORIGINAL query
                "texts": [self._doc_text(c) for c in candidates],
            }]
        }

        def call():
            import requests  # lazy
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {_gcp_token()}"},
                json=body,
                timeout=g.timeout,
            )
            if resp.status_code != 200:
                raise RerankError(f"rerank {resp.status_code}: {resp.text[:200]}")
            preds = resp.json().get("predictions")
            if not preds:
                raise RerankError("rerank returned no predictions")
            return preds[0]   # the single instance's [{"index": i, "score": s}, ...]

        try:
            data = _garden_retry(call, g.max_retries + 2)
        except RerankError:
            raise
        except Exception as e:
            raise RerankError(f"garden cross-encoder failed: {e}") from e

        # Predictions are [{"index": i, "score": s}, ...] (sorted); realign to input.
        scores = [0.0] * len(candidates)
        for item in data if isinstance(data, list) else []:
            try:
                idx = int(item["index"])
                if 0 <= idx < len(candidates):
                    scores[idx] = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
        return scores


# ---------------------------------------------------------------------------
# TIER 2 — LLM chain-of-thought collective reranker
# ---------------------------------------------------------------------------

class GardenCoTReranker(GardenReranker):
    """
    TIER 2 — Gemma 3 collective chain-of-thought rerank over the (already
    cross-encoder-filtered) shortlist. Reuses GardenReranker._score (whole-pool,
    single call; on total failure it preserves input order = the tier-1 order)
    and only swaps in the CoT prompt. Uses the ORIGINAL query. A parse failure
    keeps the tier-1 order (acceptable: tier 1 is already a strong cross-encoder).
    """

    def _llm_scores(self, question: str, candidates: list[dict]) -> dict[int, float] | None:
        g = self.config.garden
        user = format_cot_rerank_user(question, candidates, snippet=RERANK_SNIPPET)

        def call() -> str:
            client = _garden_client(g, g.llm_endpoint_id)
            resp = client.chat.completions.create(
                model=g.llm_model,
                messages=[
                    {"role": "system", "content": RERANK_COT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=self.config.reranker.temperature,
                # CoT reasoning + the final JSON array need more room than a bare
                # score array; 15 items × reasoning fits comfortably.
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""

        try:
            raw = _garden_retry(call, g.max_retries + 2)
        except Exception:
            log.warning("CoT rerank call failed — preserving tier-1 order")
            return None
        scores = parse_rerank_scores(raw)
        if scores is None:
            log.warning("CoT rerank parse failed — preserving tier-1 order")
        return scores


# ---------------------------------------------------------------------------
# Composite two-tier reranker (presents the standard reranker interface)
# ---------------------------------------------------------------------------

class GardenTwoTierReranker:
    """
    Garden reranker = TIER 1 (BGE cross-encoder: fused pool → top_k_rerank_ce)
    then TIER 2 (Gemma CoT: → top_k_rerank). Exposes the SAME
    `rerank(question, candidates, top_k)` interface as LegalReranker so the
    backend-agnostic pipeline stays unchanged — both tiers receive the ORIGINAL
    question that the pipeline already passes to step_rerank (never the rewritten
    query). If cot_enabled is False, the cross-encoder order is used directly.
    """

    def __init__(self, config: RetrievalConfig, mock: bool = False) -> None:
        self.config = config
        self.mock = mock
        self._model = None   # duck-type parity with LegalReranker (lazy-init checks)
        self.ce = GardenCrossEncoderReranker(config, mock=mock)
        self.cot = GardenCoTReranker(config, mock=mock)

    def rerank(self, question: str, candidates: list[dict],
               top_k: int | None = None) -> list[dict]:
        if not candidates:
            return []
        k_final = top_k if top_k is not None else self.config.retrieval.top_k_rerank
        tier1 = self.ce.rerank(
            question, candidates, top_k=self.config.retrieval.top_k_rerank_ce
        )
        if self.config.reranker.cot_enabled:
            return self.cot.rerank(question, tier1, top_k=k_final)
        return tier1[:k_final]


# ---------------------------------------------------------------------------
# Factory — codebase-native equivalent of the TIP's create_backends()
# ---------------------------------------------------------------------------

def make_garden_components(config: RetrievalConfig, mock: bool = False):
    """
    Build (embedder, rewriter, reranker, generator) — all PURE Garden. The
    reranker is the two-tier composite (BGE cross-encoder → Gemma CoT). Returns
    the same 4-component tuple shape the pipeline's other backends use, so the
    pipeline wiring is identical across backends.
    """
    return (
        GardenEmbedder(config, dry_run=mock),
        GardenQueryRewriter(config, mock=mock),
        GardenTwoTierReranker(config, mock=mock),
        GardenGenerator(config, mock=mock),
    )
