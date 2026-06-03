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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure legal-ai/ is importable as a package root when run as script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generator.llm_generator import LegalGenerator
from generator.prompt_builder import PromptBuilder
from retrieval.bm25_index import BM25Index
from retrieval.config import RetrievalConfig, load_config
from retrieval.embedder import LegalEmbedder
from retrieval.hybrid_search import rrf_fusion
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import LegalReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_CFG    = _ROOT / "retrieval" / "config.yaml"
_BM25_PKL       = _ROOT / "retrieval" / "data" / "bm25_index.pkl"
_CORPUS_JSON    = _ROOT / "corpus" / "data" / "corpus.json"


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
    fused_results:     list      = field(default_factory=list)
    reranked_results:  list      = field(default_factory=list)
    answer:            str       = ""
    submission_entry:  dict      = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock Qdrant (used when mock=True)
# ---------------------------------------------------------------------------

class _MockQdrant:
    """Drop-in no-op Qdrant client for offline mode."""
    def search(self, collection_name: str, query_vector: list, limit: int) -> list:
        return []
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
    ) -> None:
        self.mock = mock
        self.config: RetrievalConfig = load_config(config_path or _DEFAULT_CFG)

        self._bm25 = self._init_bm25()
        (self._embedder, self._rewriter,
         self._reranker, self._generator) = self._make_components()
        self._qdrant = _MockQdrant() if mock else self._init_qdrant()

    def _make_components(self):
        """Build the neural components for the configured backend (vllm | vertex_ai)."""
        if self.config.backend == "vertex_ai":
            # Vertex subclasses override only the model-call hook; imported here
            # so the vLLM path never touches google-genai.
            from vertex_backends import (
                VertexEmbedder, VertexGenerator, VertexQueryRewriter, VertexReranker,
            )
            return (
                VertexEmbedder(self.config, dry_run=self.mock),
                VertexQueryRewriter(self.config, mock=self.mock),
                VertexReranker(self.config, mock=self.mock),
                VertexGenerator(self.config, mock=self.mock),
            )
        return (
            LegalEmbedder(self.config, dry_run=self.mock),
            QueryRewriter(self.config, mock=self.mock),
            LegalReranker(self.config, mock=self.mock),
            LegalGenerator(self.config, mock=self.mock),
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

    def step_retrieve(self, state: PipelineState) -> PipelineState:
        """Step 2: BM25 + dense search → RRF fusion."""
        query = state.rewritten_query or state.question

        # BM25
        bm25_hits = self._bm25.search(query, top_k=self.config.retrieval.top_k_bm25)

        # Dense (rewritten query)
        q_vec       = self._embedder.embed_query(query)
        dense_hits  = self._dense_search(q_vec)

        # Dense (topic description) if non-trivial
        rankings = [bm25_hits, dense_hits]
        td = state.topic_description
        if td and td != query:
            td_vec  = self._embedder.embed_query(td)
            td_hits = self._dense_search(td_vec)
            rankings.append(td_hits)

        fused = rrf_fusion(rankings, k=self.config.retrieval.rrf_k)
        state.fused_results = self._resolve_payloads(
            fused[: self.config.retrieval.top_k_fusion]
        )
        log.debug("Retrieved %d candidates", len(state.fused_results))
        return state

    def step_rerank(self, state: PipelineState) -> PipelineState:
        """Step 3: neural reranking of fused candidates."""
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
        """Run the full pipeline for a single question. Returns submission dict."""
        state = PipelineState(question_id=q_id, question=question)
        state = self.step_rewrite(state)
        state = self.step_retrieve(state)
        state = self.step_rerank(state)
        state = self.step_generate(state)
        state = self.step_format(state)

        # Self-validate
        mentions = self._generator.extract_dieu_mentions(state.answer)
        if not mentions:
            log.warning("Q%s: answer thiếu trích dẫn 'Điều X'", q_id)

        return state.submission_entry

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
        from qdrant_client import QdrantClient  # noqa: PLC0415
        q = self.config.qdrant
        if q.url:   # Qdrant Cloud (Vertex backend)
            return QdrantClient(url=q.url, api_key=q.api_key)
        return QdrantClient(q.host, port=q.port)

    def _dense_search(self, vector) -> list[tuple[str, float]]:
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        results  = self._qdrant.search(
            collection_name=self.config.qdrant.collection,
            query_vector=vec_list,
            limit=self.config.retrieval.top_k_dense,
        )
        out = []
        for r in results:
            if hasattr(r, "id") and hasattr(r, "score"):
                out.append((str(r.id), float(r.score)))
            elif isinstance(r, dict):
                out.append((str(r["id"]), float(r.get("score", 0.0))))
        return out

    def _resolve_payloads(self, fused: list[tuple[str, float]]) -> list[dict]:
        results = []
        for chunk_id, rrf_score in fused:
            payload = self._bm25.get_payload(chunk_id)
            if payload is None:
                payload = {"chunk_id": chunk_id}
            results.append({**payload, "rrf_score": rrf_score})
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LegalAI pipeline runner")
    parser.add_argument("--config", default=None,
                        help="Config YAML (default: retrieval/config.yaml; "
                             "use config_vertex.yaml for the Vertex AI backend)")
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
