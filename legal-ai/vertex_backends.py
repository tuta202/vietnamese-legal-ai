"""
vertex_backends.py — Vertex AI (Gemini) implementations of the four neural
components, as subclasses that override ONLY the model-call hook.

Everything else (prompts, few-shots, JSON parsing, RRF fusion, reranking flow,
submission formatting) is inherited unchanged from the vLLM base classes in
retrieval/ and generator/. Switching backends therefore changes only the model.

  QueryRewriter   → VertexQueryRewriter   (overrides _chat_complete)
  LegalGenerator  → VertexGenerator       (overrides _chat_complete)
  LegalEmbedder   → VertexEmbedder        (overrides _encode_query/_encode_documents)
  LegalReranker   → VertexReranker        (overrides _score)

Heavy SDKs (google.genai) are imported lazily inside methods, so importing this
module — or constructing any class with mock/dry_run=True — never touches the SDK.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from generator.llm_generator import LegalGenerator
from retrieval.config import RetrievalConfig
from retrieval.embedder import LegalEmbedder
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import LegalReranker


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_genai_client(config: RetrievalConfig):
    """
    Build a google-genai client from config.vllm.{gcp_project,gcp_location}.
      • project set   → Vertex AI mode
      • project empty → GOOGLE_API_KEY env var (Gemini API mode)
    """
    from google import genai  # lazy

    project  = (config.vllm.gcp_project or "").strip()
    location = (config.vllm.gcp_location or "us-central1").strip()
    if project:
        return genai.Client(vertexai=True, project=project, location=location)
    return genai.Client()


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _retry(fn, *, attempts: int = 6, base: float = 8.0):
    """
    Call fn(); on transient Gemini errors (429 RESOURCE_EXHAUSTED, 5xx) retry
    with exponential backoff (capped 60s). Non-transient errors propagate.
    """
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # google.genai errors are not all subclasses we can import cheaply
            msg = str(e)
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            transient = (
                "429" in msg or "RESOURCE_EXHAUSTED" in msg
                or "503" in msg or "500" in msg or "UNAVAILABLE" in msg
                or (isinstance(code, int) and code in (429, 500, 503))
            )
            if not transient:
                raise
            last = e
            time.sleep(min(base * (2 ** i), 60))
    if last:
        raise last


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

class VertexQueryRewriter(QueryRewriter):
    """Gemini-backed query rewriter; inherits prompts, few-shots, parsing."""

    @property
    def _llm(self):
        if self._client is None:
            self._client = make_genai_client(self.config)
        return self._client

    def _chat_complete(self, messages: list[dict]) -> str:
        from google.genai import types  # lazy

        system = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            grole = "user" if m["role"] == "user" else "model"
            contents.append(
                types.Content(role=grole, parts=[types.Part(text=m["content"])])
            )
        resp = _retry(lambda: self._llm.models.generate_content(
            model=self.config.vllm.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=0.1,
                max_output_tokens=512,
            ),
        ))
        return getattr(resp, "text", None) or ""


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

class VertexGenerator(LegalGenerator):
    """Gemini-backed generator; inherits the 8-rule PromptBuilder + mock answer."""

    @property
    def _llm(self):
        if self._client is None:
            self._client = make_genai_client(self.config)
        return self._client

    def _chat_complete(self, system: str, user: str) -> str:
        from google.genai import types  # lazy

        gen_cfg = self.config.generator
        resp = _retry(lambda: self._llm.models.generate_content(
            model=self.config.vllm.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=gen_cfg.temperature,
                max_output_tokens=gen_cfg.max_tokens,
                top_p=gen_cfg.top_p,
            ),
        ))
        return getattr(resp, "text", None) or ""


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class VertexEmbedder(LegalEmbedder):
    """
    Gemini-embedding-001 backed embedder. Inherits embed_query/embed_documents/
    embed_corpus; overrides only the encode hooks. `dim` comes from
    config.embedding.dimension (3072 for Vertex). No Qwen "Instruct:" wrapper —
    Gemini uses task_type instead.
    """

    def __init__(self, config: RetrievalConfig, dry_run: bool = False) -> None:
        super().__init__(config, dry_run=dry_run)
        self._genai = None

    @property
    def genai_client(self):
        if self._genai is None:
            self._genai = make_genai_client(self.config)
        return self._genai

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types  # lazy

        resp = _retry(lambda: self.genai_client.models.embed_content(
            model=self.config.models.embedder,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dim,
            ),
        ))
        return [list(e.values) for e in resp.embeddings]

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self._embed([text], self.config.embedding.query_task_type)[0]
        return np.array(vec, dtype=np.float32)

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._embed(list(texts), self.config.embedding.task_type)
        return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = (
    "Bạn là chuyên gia pháp lý Việt Nam. Chấm điểm mức độ liên quan của từng "
    "điều luật đối với câu hỏi, thang điểm 0 đến 10 (10 = liên quan trực tiếp "
    "nhất). CHỈ trả về một mảng JSON, không thêm văn bản nào khác, đúng định dạng:\n"
    '[{"index": 0, "score": 8}, {"index": 1, "score": 3}]'
)
_RERANK_SNIPPET = 400


class VertexReranker(LegalReranker):
    """
    LLM-based reranker (Gemini). Overrides only the scoring hook; rerank()
    (mock short-circuit + sort + truncate) is inherited unchanged. There is no
    Gemini cross-encoder, so scoring is a single batched 0–10 LLM judgement.
    """

    def __init__(self, config: RetrievalConfig, mock: bool = False) -> None:
        super().__init__(config, mock=mock)
        self._genai = None

    @property
    def genai_client(self):
        if self._genai is None:
            self._genai = make_genai_client(self.config)
        return self._genai

    def _score(self, question: str, candidates: list[dict]) -> list[float]:
        scores_map = self._llm_scores(question, candidates)
        if not scores_map:
            # Fallback: preserve retrieval order (descending dummy scores).
            n = len(candidates)
            return [float(n - i) for i in range(n)]
        return [float(scores_map.get(i, 0.0)) for i in range(len(candidates))]

    def _llm_scores(self, question: str, candidates: list[dict]) -> dict[int, float] | None:
        from google.genai import types  # lazy

        blocks = [f"Câu hỏi: {question}", "", "Danh sách điều luật cần chấm điểm:"]
        for i, c in enumerate(candidates):
            blocks.append(
                f"[{i}] {c.get('dieu_number', '')} {c.get('dieu_title', '')}\n"
                f"{c.get('content', '')[:_RERANK_SNIPPET]}"
            )
        try:
            resp = _retry(lambda: self.genai_client.models.generate_content(
                model=self.config.models.reranker,
                contents="\n".join(blocks),
                config=types.GenerateContentConfig(
                    system_instruction=_RERANK_SYSTEM,
                    temperature=self.config.reranker.temperature,
                ),
            ))
            return self._parse_scores(getattr(resp, "text", None) or "")
        except Exception:
            return None

    @staticmethod
    def _parse_scores(raw: str) -> dict[int, float] | None:
        cleaned = _strip_code_fences(raw)
        m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if m:
            cleaned = m.group(0)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, list):
            return None
        out: dict[int, float] = {}
        for item in data:
            try:
                out[int(item["index"])] = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
        return out or None
