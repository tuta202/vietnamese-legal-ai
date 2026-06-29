"""
GPU backend — self-deployed endpoints (open-source, competition-compliant)
implementations of the four neural components, as subclasses that override ONLY
the model-call hook — exactly mirroring vertex_backends.py.

Every component runs an open-source model served on a self-deployed Vertex AI
Online Endpoint (GPU). There is no Gemini fallback here — for the Gemini stack
use `backend: vertex_ai` (config_vertex_clean.yaml) instead. Clean separation:
  • vertex_ai → Gemini everywhere
  • gpu       → self-deployed open-source models everywhere

  QueryRewriter   → GpuQueryRewriter   (overrides _chat_complete)
  LegalGenerator  → GpuGenerator       (overrides _chat_complete)
  LegalEmbedder   → GpuEmbedder        (overrides _encode_query/_encode_documents)
  LegalReranker   → GpuReranker        (overrides _score)

Endpoints are OpenAI-compatible vLLM servers:
  • Embedder  : Qwen3-Embedding-8B  → 4096-dim vectors
  • LLM/rerank: Gemma 3 12B-it      → generation + 0–10 relevance scoring

Auth is a google-auth bearer token passed as the OpenAI api_key. The token
expires after ~1h, so a fresh client (with a freshly-minted token) is built for
every call; a 401 forces a credential refresh on the next retry.

Heavy SDKs (openai, google.auth) are imported lazily inside methods, so importing
this module — or constructing any class with mock/dry_run=True — never touches
them. Shared, provider-agnostic helpers come from legal_rag.common.backends_common.
"""
from __future__ import annotations

import threading

import numpy as np

from legal_rag.common.backends_common import (
    RERANK_SYSTEM,
    build_rerank_blocks,
    parse_rerank_scores,
    retry_transient,
)
from legal_rag.generation.llm_generator import LegalGenerator
from legal_rag.retrieval.config import GpuConfig, RetrievalConfig
from legal_rag.retrieval.embedder import LegalEmbedder
from legal_rag.retrieval.query_rewriter import QueryRewriter
from legal_rag.retrieval.reranker import LegalReranker


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GpuError(Exception):
    """Base error for the GPU endpoints backend."""


class EmbeddingError(GpuError):
    """Raised when an embedding call fails after retries."""


class GenerationError(GpuError):
    """Raised when a chat-completion call fails after retries."""


class RerankError(GpuError):
    """Raised when a rerank scoring call fails after retries."""


# ---------------------------------------------------------------------------
# Auth + client + retry helpers
# ---------------------------------------------------------------------------

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_creds = None   # module-level cached ADC credentials
_creds_lock = threading.Lock()


def _gcp_token() -> str:
    """Return a valid GCP bearer token, refreshing application-default creds."""
    global _creds
    import google.auth  # lazy
    from google.auth.transport.requests import Request  # lazy

    with _creds_lock:
        if _creds is None:
            _creds, _ = google.auth.default(scopes=_SCOPES)
        if not _creds.valid:
            _creds.refresh(Request())
        return _creds.token


def _reset_creds() -> None:
    """Drop the cached credentials so the next token() mints a fresh one."""
    global _creds
    with _creds_lock:
        _creds = None


def _gpu_client(g: GpuConfig, endpoint_id: str):
    """
    Build an OpenAI client for the Gpu LLM endpoint. A *dedicated* endpoint
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


def _gpu_retry(fn, attempts: int):
    """Gpu retry: transient backoff + bearer-token refresh on 401."""
    return retry_transient(fn, attempts=attempts, base=4.0, refresh_auth=_reset_creds)


def gpu_chat_complete(
    config: RetrievalConfig,
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Call the configured Gemma endpoint for provider-neutral LLM stages."""
    g = config.gpu

    def call() -> str:
        client = _gpu_client(g, g.llm_endpoint_id)
        response = client.chat.completions.create(
            model=g.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    try:
        return _gpu_retry(call, g.max_retries + 2)
    except Exception as exc:
        raise GenerationError(f"gpu chat completion failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

class GpuQueryRewriter(QueryRewriter):
    """Gemma-backed query rewriter; inherits prompts, few-shots, JSON parsing."""

    def _chat_complete(self, messages: list[dict]) -> str:
        g = self.config.gpu

        def call() -> str:
            client = _gpu_client(g, g.llm_endpoint_id)
            resp = client.chat.completions.create(
                model=g.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""

        try:
            return _gpu_retry(call, g.max_retries + 2)
        except Exception as e:
            raise GenerationError(f"gpu rewrite failed: {e}") from e


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

class GpuGenerator(LegalGenerator):
    """Gemma-backed generator; inherits the PromptBuilder rules + mock answer."""

    def _chat_complete(self, system: str, user: str) -> str:
        g = self.config.gpu
        gen = self.config.generator

        def call() -> str:
            client = _gpu_client(g, g.llm_endpoint_id)
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
            return _gpu_retry(call, g.max_retries + 2)
        except Exception as e:
            raise GenerationError(f"gpu generate failed: {e}") from e


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class GpuEmbedder(LegalEmbedder):
    """
    Qwen3-Embedding-8B backed embedder (OpenAI-compatible Gpu endpoint).
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
        g = self.config.gpu
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
                # Surface the status code so _gpu_retry can classify 401/429/5xx.
                raise EmbeddingError(f"predict {resp.status_code}: {resp.text[:200]}")
            preds = resp.json().get("predictions")
            if not preds:
                raise EmbeddingError("predict returned no predictions")
            # Each prediction is [[...vector...]]; unwrap the inner vector.
            return [p[0] if (p and isinstance(p[0], list)) else p for p in preds]

        try:
            return _gpu_retry(call, g.max_retries + 2)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"gpu embed failed: {e}") from e

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


class GpuReranker(LegalReranker):
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
        g = self.config.gpu
        user_msg = build_rerank_blocks(question, candidates)

        def call() -> str:
            client = _gpu_client(g, g.llm_endpoint_id)
            resp = client.chat.completions.create(
                model=g.llm_model,
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.config.reranker.temperature,
                # Bound the JSON array. Whole-pool scoring (~40 items) needs more
                # room than a 10-item batch did (~40×{"index":N,"score":N}).
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""

        try:
            raw = _gpu_retry(call, g.max_retries + 2)
        except Exception:
            return None
        return parse_rerank_scores(raw)


# ---------------------------------------------------------------------------
# Factory — codebase-native equivalent of the TIP's create_backends()
# ---------------------------------------------------------------------------

def make_gpu_components(config: RetrievalConfig, mock: bool = False):
    """
    Build (embedder, rewriter, reranker, generator) — all self-deployed GPU
    endpoints. Returns the same 4-component tuple the pipeline's other backend uses.
    """
    return (
        GpuEmbedder(config, dry_run=mock),
        GpuQueryRewriter(config, mock=mock),
        GpuReranker(config, mock=mock),
        GpuGenerator(config, mock=mock),
    )
