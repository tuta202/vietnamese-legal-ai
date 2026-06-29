"""
LegalAIPipeline — end-to-end orchestrator connecting retrieval + generation.

CLI:
    python pipeline.py --mock --input tests/sample_questions.json --output results.json

Each step method (step_rewrite, step_retrieve, step_rerank, step_generate,
step_format) is designed to wrap directly as a LangGraph v2 node.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from legal_rag.generation.prompt_builder import _DISCLAIMER
from legal_rag.common.paths import CORPUS_DATA_DIR, INDEX_DATA_DIR, PROJECT_ROOT
from legal_rag.common.qdrant import create_qdrant_client
from legal_rag.retrieval.bm25_index import BM25Index
from legal_rag.retrieval.config import RetrievalConfig, load_config, validate_config
from legal_rag.retrieval.hybrid_search import rrf_fusion
from legal_rag.retrieval.intent_decomposer import LegalIntentDecomposer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_CFG    = PROJECT_ROOT / "config_gpu_clean.yaml"
_BM25_PKL       = INDEX_DATA_DIR / "bm25_index.pkl"
_CORPUS_JSON    = CORPUS_DATA_DIR / "corpus.json"


def _norm_article(value: str) -> str:
    return " ".join(str(value or "").split())


def _doc_from_article(value: str) -> str:
    parts = _norm_article(value).split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Immutable-ish state object; each step enriches it and returns it."""
    question_id:       int | str
    question:          str
    rewritten_query:   str       = ""
    topic_description: str       = ""
    intent_queries:    list[str] = field(default_factory=list)
    fused_results:     list      = field(default_factory=list)
    intent_hits:       list      = field(default_factory=list)
    retrieval_metrics: dict      = field(default_factory=dict)
    expected_articles: list[str] = field(default_factory=list)
    expected_docs:     list[str] = field(default_factory=list)
    reranked_results:  list      = field(default_factory=list)
    answer:            str       = ""
    submission_entry:  dict      = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock Qdrant (used when mock=True)
# ---------------------------------------------------------------------------

class _MockQdrant:
    """Drop-in no-op Qdrant client for offline mode."""
    def query_points(self, collection_name: str, query: list, limit: int,
                     with_payload: bool = True):
        return SimpleNamespace(points=[])
    def retrieve(self, collection_name: str, ids: list, with_payload: bool) -> list:
        return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LegalAIPipeline:
    """
    Orchestrates: query rewrite → BM25+dense retrieval → rerank → generate.

    mock=True: runs the full flow without GPU/Docker/vLLM.
    BM25 is always real; all neural models use mock/dry_run flags.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        mock: bool = False,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.mock = mock
        self.config: RetrievalConfig = load_config(config_path or _DEFAULT_CFG)

        # Fail fast on a real run with an incomplete config (skip in mock).
        if not mock:
            problems = validate_config(self.config)
            if problems:
                raise ValueError(
                    f"Invalid config for backend '{self.config.backend}': "
                    + "; ".join(problems)
                )

        self._bm25 = bm25_index if bm25_index is not None else self._init_bm25()
        (self._embedder, self._rewriter,
         self._reranker, self._generator) = self._make_components()
        self._decomposer = LegalIntentDecomposer(
            self.config,
            mock=self.mock,
            chat_complete=getattr(self._rewriter, "_chat_complete", None),
        )
        self._qdrant = _MockQdrant() if mock else self._init_qdrant()

        # Optional detailed score logging for diagnostic callers.
        self._scores_detail_path: str | None = None
        self._scores_detail_lock = threading.Lock()

    def _make_components(self):
        """
        Build the four neural components for the configured backend.

        Two backends are supported, behaving identically and differing only in
        the underlying models (see config_gpu_clean.yaml / config_vertex_clean.yaml):
          • gpu    → self-deployed Qwen3 + Gemma on GPU endpoints
          • vertex_ai → Gemini
        Both subclass the shared base components (LegalEmbedder/QueryRewriter/
        LegalReranker/LegalGenerator), overriding only the model-call hook.
        """
        if self.config.backend == "gpu":
            from legal_rag.backends.gpu import make_gpu_components
            return make_gpu_components(self.config, mock=self.mock)
        if self.config.backend == "vertex_ai":
            # Imported lazily so the gpu path never touches google-genai.
            from legal_rag.backends.vertex import (
                VertexEmbedder, VertexGenerator, VertexQueryRewriter, VertexReranker,
            )
            return (
                VertexEmbedder(self.config, dry_run=self.mock),
                VertexQueryRewriter(self.config, mock=self.mock),
                VertexReranker(self.config, mock=self.mock),
                VertexGenerator(self.config, mock=self.mock),
            )
        raise ValueError(
            f"unknown backend {self.config.backend!r}; expected 'gpu' or 'vertex_ai'"
        )

    # ------------------------------------------------------------------
    # Step methods (LangGraph v2 node-ready)
    # ------------------------------------------------------------------

    def step_rewrite(self, state: PipelineState) -> PipelineState:
        """Step 1: rewrite question into optimised query + topic description."""
        result = self._rewriter.rewrite(state.question)
        state.rewritten_query   = result["rewritten_query"]
        state.topic_description = result["topic_description"]
        log.debug("Rewritten: %s", state.rewritten_query[:80])
        return state

    def step_decompose(self, state: PipelineState) -> PipelineState:
        """Step 1b: decompose the original question into legal retrieval intents."""
        analysis = self._decomposer.decompose(state.question)
        state.intent_queries = analysis.intents
        log.debug("Decomposed into %d intent(s)", len(state.intent_queries))
        return state

    def step_retrieve(self, state: PipelineState) -> PipelineState:
        """Step 2: BM25 + dense search → RRF fusion."""
        query = state.rewritten_query or state.question

        # BM25
        bm25_hits = self._bm25.search(query, top_k=self.config.retrieval.top_k_bm25)

        # Dense (rewritten query) — returns (hits, payload-by-chunk_id)
        q_vec       = self._embedder.embed_query(query)
        dense_hits, dense_payloads = self._dense_search(q_vec)

        # Dense (topic description) if non-trivial
        rankings = [bm25_hits, dense_hits]
        td = state.topic_description
        if td and td != query:
            td_vec  = self._embedder.embed_query(td)
            td_hits, td_payloads = self._dense_search(td_vec)
            rankings.append(td_hits)
            dense_payloads = {**dense_payloads, **td_payloads}

        fused = rrf_fusion(rankings, k=self.config.retrieval.rrf_k)
        global_results = self._resolve_payloads(
            fused[: self.config.retrieval.top_k_fusion], dense_payloads
        )
        for r in global_results:
            r.setdefault("retrieval_source", "global")
            r.setdefault("from_intent", False)
            r.setdefault("intent_ids", [])

        # Fallback: if nothing resolved (e.g. rewritten query missed), retry BM25
        # on the original question so we never produce an empty-context answer.
        if not global_results:
            fb = self._bm25.search(state.question, top_k=self.config.retrieval.top_k_fusion)
            global_results = self._resolve_payloads(fb, {})
            for r in global_results:
                r.setdefault("retrieval_source", "global_fallback")
                r.setdefault("from_intent", False)
                r.setdefault("intent_ids", [])

        intent_results = self._retrieve_intent_hits(state)
        state.intent_hits = intent_results

        state.fused_results = global_results
        state.retrieval_metrics = self._retrieval_metrics(state, global_results, intent_results)
        log.debug(
            "Retrieved %d global candidates plus %d intent hits from %d intent(s)",
            len(state.fused_results),
            len(intent_results),
            len(state.intent_queries),
        )
        return state

    def step_rerank(self, state: PipelineState) -> PipelineState:
        """Step 3: neural reranking of fused candidates."""
        if self._scores_detail_path and hasattr(self._reranker, "rerank_with_scores"):
            top_results, full_pool = self._reranker.rerank_with_scores(
                state.question,
                state.fused_results,
                top_k=self.config.retrieval.top_k_rerank,
            )
            state.reranked_results = top_results
            record = {
                "id": state.question_id,
                "question": state.question,
                "rewritten_query": state.rewritten_query,
                "topic_description": state.topic_description,
                "intents": state.intent_queries,
                "num_intents": len(state.intent_queries),
                "retrieval_metrics": state.retrieval_metrics,
                "pool_size": len(state.fused_results),
                "intent_hit_count": len(state.intent_hits),
                "keep_count": self._keep_count_for_logging(state.fused_results, state.intent_hits),
                "candidates": full_pool,
                "final_count": len(top_results),
                "final_articles": [r.get("relevant_article_str", "") for r in top_results],
            }
            self._append_detail(record)
        else:
            state.reranked_results = self._reranker.rerank(
                state.question,
                state.fused_results,
                top_k=self.config.retrieval.top_k_rerank,
            )
        log.debug("Reranked to %d articles", len(state.reranked_results))
        return state

    def step_generate(self, state: PipelineState) -> PipelineState:
        """Step 4: LLM generation grounded in reranked articles."""
        state.answer = self._generator.generate(
            state.question, state.reranked_results
        )
        return state

    def step_format(self, state: PipelineState) -> PipelineState:
        """Step 5: build submission entry with deduplicated references."""
        articles = state.reranked_results

        seen_docs, rel_docs = set(), []
        for art in articles:
            ds = art.get("relevant_doc_str", "")
            if ds and ds not in seen_docs:
                seen_docs.add(ds)
                rel_docs.append(ds)

        seen_arts, rel_arts = set(), []
        for art in articles:
            as_ = art.get("relevant_article_str", "")
            if as_ and as_ not in seen_arts:
                seen_arts.add(as_)
                rel_arts.append(as_)

        state.submission_entry = {
            "id":                 state.question_id,
            "question":           state.question,
            "answer":             state.answer,
            "relevant_docs":      rel_docs,
            "relevant_articles":  rel_arts,
        }
        return state

    # ------------------------------------------------------------------
    # Orchestrators
    # ------------------------------------------------------------------

    def process_question(self, q_id: int | str, question: str) -> dict:
        """
        Run the full pipeline for a single question. Returns a submission dict.

        Failures are isolated: any exception is logged and a well-formed (5-field)
        fallback entry is returned so one bad question never aborts the batch.
        """
        try:
            state = PipelineState(question_id=q_id, question=question)
            state = self.step_rewrite(state)
            state = self.step_decompose(state)
            state = self.step_retrieve(state)
            state = self.step_rerank(state)
            state = self.step_generate(state)
            state = self.step_format(state)

            # Self-validate
            mentions = self._generator.extract_dieu_mentions(state.answer)
            if not mentions:
                log.warning("Q%s: answer thiếu trích dẫn 'Điều X'", q_id)

            return state.submission_entry
        except Exception:
            log.exception("Q%s failed — emitting fallback entry", q_id)
            return {
                "id":                q_id,
                "question":          question,
                "answer":            _DISCLAIMER,
                "relevant_docs":     [],
                "relevant_articles": [],
            }

    def process_question_retrieve_only(self, q_id: int | str, question: str) -> dict:
        """
        Run ONLY the retrieval stage (rewrite → BM25+dense → RRF) and stop after
        fusion — no rerank, no generate. The submission entry is built from the
        top `top_k_fusion` fused candidates with an EMPTY answer field. Used to
        measure recall@K (e.g. recall@100) of the retrieval funnel.

        Same fail-soft contract as process_question: any error → fallback entry.
        """
        try:
            state = PipelineState(question_id=q_id, question=question)
            state = self.step_rewrite(state)
            state = self.step_decompose(state)
            state = self.step_retrieve(state)
            # top_k_fusion already truncates fused_results; feed straight to format.
            state.reranked_results = state.fused_results
            state.answer = ""
            state = self.step_format(state)
            return state.submission_entry
        except Exception:
            log.exception("Q%s failed (retrieve-only) — emitting fallback entry", q_id)
            return {
                "id":                q_id,
                "question":          question,
                "answer":            "",
                "relevant_docs":     [],
                "relevant_articles": [],
            }

    def process_questions(self, questions: list[dict]) -> list[dict]:
        """Process a list of {id, question} dicts."""
        results = []
        for item in questions:
            q_id     = item["id"]
            question = item["question"]
            log.info("Processing Q%s: %s…", q_id, question[:60])
            results.append(self.process_question(q_id, question))
        return results

    def process_batch(self, questions_file: Path | str) -> list[dict]:
        """Load questions from JSON file and process each one."""
        questions = json.loads(
            Path(questions_file).read_text(encoding="utf-8")
        )
        return self.process_questions(questions)

    def _append_detail(self, record: dict) -> None:
        """Thread-safe append of one score-detail record to the JSONL file."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._scores_detail_lock:
            with open(self._scores_detail_path, "a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_bm25(self) -> BM25Index:
        if _BM25_PKL.exists():
            try:
                log.info("Loading BM25 index from %s", _BM25_PKL)
                return BM25Index.load(_BM25_PKL)
            except (AttributeError, ModuleNotFoundError) as e:
                # pkl was created with __name__ == '__main__' (CLI run) — class
                # path mismatch when loading from a module context. Rebuild.
                log.warning("BM25 pickle stale (%s) — rebuilding from corpus", e)
                _BM25_PKL.unlink(missing_ok=True)
        if _CORPUS_JSON.exists():
            log.info("Building BM25 index from corpus…")
            articles = json.loads(_CORPUS_JSON.read_text(encoding="utf-8"))["articles"]
            idx = BM25Index().build_from_corpus(articles)
            _BM25_PKL.parent.mkdir(parents=True, exist_ok=True)
            idx.save(_BM25_PKL)
            return idx
        log.warning("No corpus found — starting with empty BM25 index")
        return BM25Index()

    def _init_qdrant(self) -> Any:
        return create_qdrant_client(self.config)

    def _retrieve_intent_hits(self, state: PipelineState) -> list[dict]:
        """Retrieve top intent candidates using BM25(intent) + Dense(intent)."""
        cfg = self.config.retrieval
        if not cfg.enable_intent_retrieval or not state.intent_queries:
            return []

        by_chunk: dict[str, dict] = {}
        for intent_id, intent in enumerate(state.intent_queries, start=1):
            text = intent.strip()
            if not text:
                continue
            bm25_hits = self._bm25.search(text, top_k=cfg.intent_top_k_bm25)
            vec = self._embedder.embed_query(text)
            dense_hits, dense_payloads = self._dense_search(vec, limit=cfg.intent_top_k_dense)
            fused = rrf_fusion([bm25_hits, dense_hits], k=cfg.rrf_k)[: cfg.intent_top_k_rrf]
            resolved = self._resolve_payloads(fused, dense_payloads)
            for rank, item in enumerate(resolved, start=1):
                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    continue
                current = by_chunk.get(chunk_id)
                if current is None:
                    item["retrieval_source"] = "intent"
                    item["from_intent"] = True
                    item["intent_ids"] = [intent_id]
                    item["intent_queries"] = [text]
                    item["intent_rank"] = rank
                    item["intent_rrf_score"] = item.get("rrf_score", 0.0)
                    by_chunk[chunk_id] = item
                else:
                    ids = current.setdefault("intent_ids", [])
                    if intent_id not in ids:
                        ids.append(intent_id)
                    queries = current.setdefault("intent_queries", [])
                    if text not in queries:
                        queries.append(text)
                    current["intent_rank"] = min(current.get("intent_rank", rank), rank)

        return sorted(
            by_chunk.values(),
            key=lambda c: (c.get("intent_rank", 10**9), -float(c.get("intent_rrf_score", 0.0))),
        )

    def _retrieval_metrics(
        self,
        state: PipelineState,
        global_results: list[dict],
        intent_results: list[dict],
    ) -> dict:
        """Per-question retrieval diagnostics; recall is filled when gold is present."""
        expected = {_norm_article(a) for a in state.expected_articles if a}

        def _article_set(items: list[dict]) -> set[str]:
            return {_norm_article(x.get("relevant_article_str", "")) for x in items if x.get("relevant_article_str")}

        global_set = _article_set(global_results)
        intent_set = _article_set(intent_results)
        keep_set = global_set | intent_set

        def _recall(found: set[str]) -> float | None:
            if not expected:
                return None
            return len(found & expected) / len(expected)

        expected_docs = {_norm_article(d) for d in state.expected_docs if d}
        if not expected_docs and expected:
            expected_docs = {_doc_from_article(a) for a in expected if _doc_from_article(a)}
        cross_document = len(expected_docs) > 1 if expected_docs else None

        return {
            "num_intents": len(state.intent_queries),
            "intent_lengths": [len(i) for i in state.intent_queries],
            "global_count": len(global_results),
            "intent_count": len(intent_results),
            "keep_count": len(keep_set),
            "global_recall": _recall(global_set),
            "intent_only_recall": _recall(intent_set),
            "keep_recall": _recall(keep_set),
            "cross_document": cross_document,
        }

    @staticmethod
    def _keep_count_for_logging(global_results: list[dict], intent_results: list[dict]) -> int:
        articles = {
            _norm_article(c.get("relevant_article_str", ""))
            for c in global_results + intent_results
            if c.get("relevant_article_str")
        }
        return len(articles)

    def _dense_search(
        self,
        vector,
        limit: int | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, dict]]:
        """
        Dense search via Qdrant query_points.

        Returns (hits, payloads):
          • hits     = [(chunk_id, score), ...] keyed by the RAW chunk_id from the
            Qdrant payload — so fusion + BM25 payload lookup line up (this is the
            fix for dense-only hits previously keyed by the dashed point UUID).
          • payloads = {chunk_id: qdrant_payload} for resolving hits not in BM25.
        """
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        
        import time
        max_attempts = 5
        backoff = 1.0
        resp = None

        for attempt in range(max_attempts):
            try:
                resp = self._qdrant.query_points(
                    collection_name=self.config.qdrant.collection,
                    query=vec_list,
                    limit=limit or self.config.retrieval.top_k_dense,
                    with_payload=True,
                )
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    log.error("Qdrant search failed after %d attempts", max_attempts)
                    raise e
                log.warning(
                    "Qdrant query failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, max_attempts, e, backoff
                )
                time.sleep(backoff)
                backoff *= 2.0

        points = getattr(resp, "points", resp)

        hits: list[tuple[str, float]] = []
        payloads: dict[str, dict] = {}
        for r in points:
            payload = getattr(r, "payload", None)
            if payload is None and isinstance(r, dict):
                payload = r.get("payload")
            payload = payload or {}

            rid = getattr(r, "id", None)
            if rid is None and isinstance(r, dict):
                rid = r.get("id")
            chunk_id = payload.get("chunk_id") or str(rid)

            score = getattr(r, "score", None)
            if score is None and isinstance(r, dict):
                score = r.get("score", 0.0)

            hits.append((chunk_id, float(score or 0.0)))
            if payload:
                payloads[chunk_id] = payload
        return hits, payloads

    def _resolve_payloads(
        self,
        fused: list[tuple[str, float]],
        dense_payloads: dict[str, dict] | None = None,
    ) -> list[dict]:
        dense_payloads = dense_payloads or {}
        results = []
        for chunk_id, rrf_score in fused:
            payload = self._bm25.get_payload(chunk_id)
            if payload is None:
                qp = dense_payloads.get(chunk_id)
                payload = self._normalize_payload(qp, chunk_id) if qp else {"chunk_id": chunk_id}
            results.append({**payload, "rrf_score": rrf_score})
        return results

    @staticmethod
    def _normalize_payload(qp: dict, chunk_id: str) -> dict:
        """Qdrant payload → article dict shape with submission strings guaranteed."""
        law_id   = qp.get("law_id", "")
        law_type = qp.get("law_type", "")
        law_name = qp.get("law_name", "")
        dieu_number = qp.get("dieu_number") or qp.get("article_number", "")
        dieu_title  = qp.get("dieu_title")  or qp.get("article_title", "")
        return {
            "chunk_id":             qp.get("chunk_id", chunk_id),
            "law_id":               law_id,
            "law_type":             law_type,
            "law_name":             law_name,
            "dieu_number":          dieu_number,
            "dieu_title":           dieu_title,
            "content":              qp.get("content", ""),
            "relevant_doc_str":     qp.get("relevant_doc_str") or f"{law_id}|{law_type} {law_name}",
            "relevant_article_str": qp.get("relevant_article_str")
                                    or f"{law_id}|{law_type} {law_name}|{dieu_number}",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LegalAI pipeline runner")
    parser.add_argument("--config", default=None,
                        help="Config YAML (default: config_gpu_clean.yaml; "
                             "use config_vertex_clean.yaml for the Gemini backend)")
    parser.add_argument("--mock",   action="store_true", help="Run in mock mode (no GPU/Docker)")
    parser.add_argument("--input",  required=True, help="JSON file with [{id, question}, ...]")
    parser.add_argument("--output", default="results.json", help="Output results.json path")
    args = parser.parse_args()

    pipeline = LegalAIPipeline(config_path=args.config, mock=args.mock)
    results  = pipeline.process_batch(args.input)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote %d results → %s", len(results), out_path)


if __name__ == "__main__":
    main()
