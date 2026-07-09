from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from legal_rag.backends.bge import BgeScorer
from legal_rag.common.article_lookup import ArticleLookup
from legal_rag.generation.generate_answers import generate_one, generation_signature
from legal_rag.output.submission import save_submission
from legal_rag.pipeline import LegalAIPipeline
from legal_rag.production.query_router import (
    LEGAL_ROUTE,
    SMALL_TALK_ROUTE,
    answer_small_talk,
    out_of_scope_answer,
    route_query,
)
from legal_rag.ranking.candidate_union import build_entry
from legal_rag.ranking.intent_bge import load_corpus, rank_one_intent
from legal_rag.retrieval.bm25_index import BM25Index
from legal_rag.retrieval.config import load_config, validate_config
from legal_rag.retrieval.hybrid_search import rrf_fusion
from legal_rag.retrieval.intent_rrf import retrieve_ranked_hits_for_question
from legal_rag.retrieval.query_analysis import _decompose_one, _rewrite_one
from legal_rag.verification.candidate_verifier import verify_one
from legal_rag.verification.deterministic_cleanup import should_drop_penalty as stage1_should_drop_penalty
from legal_rag.verification.enforcement_gate import drop_reason as enforcement_drop_reason
from legal_rag.verification.enforcement_gate import is_coercion_article, is_penalty_article
from legal_rag.verification.final_collective import SYSTEM_PROMPT as FINAL_COLLECTIVE_SYSTEM_PROMPT
from legal_rag.verification.final_collective import process_question as final_collective_question
from legal_rag.verification.intent_coverage_rescue import rescue_question


log = logging.getLogger("legal_rag.production.single_question_runner")


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = " ".join(str(item or "").split())
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def doc_ref(article_ref: str) -> str:
    parts = str(article_ref).split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else str(article_ref)


def to_submission_row(qid: str | int, question: str, articles: list[str], answer: str = "") -> dict:
    articles = dedupe_keep_order(articles)
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "relevant_docs": dedupe_keep_order([doc_ref(article) for article in articles]),
        "relevant_articles": articles,
    }


class SingleQuestionRagRunner:
    """Low-latency, in-process version of the best RAG workflow.

    The object is intended to be created once at service startup. Model clients,
    BM25, Qdrant, corpus lookup, and BGE scorer are then reused across requests.
    """

    def __init__(
        self,
        *,
        config_path: str | Path = "config_gpu_gemini_production.yaml",
        corpus_path: str | Path = "corpus/data/corpus_clean_asof_20260301.json",
        bm25_index_path: str | Path = "retrieval/data/bm25_index_asof_20260301.pkl",
        expected_articles: int = 82570,
        stage1_batch_workers: int = 2,
        final_batch_workers: int = 2,
        bge_workers: int = 4,
        rescue_coverage_depth: int = 4,
    ) -> None:
        self.config_path = Path(config_path)
        self.corpus_path = Path(corpus_path)
        self.bm25_index_path = Path(bm25_index_path)
        self.stage1_batch_workers = max(1, stage1_batch_workers)
        self.final_batch_workers = max(1, final_batch_workers)
        self.bge_workers = max(1, bge_workers)
        self.rescue_coverage_depth = rescue_coverage_depth

        self.config = load_config(self.config_path)
        problems = validate_config(self.config)
        if problems:
            raise ValueError("Invalid config: " + "; ".join(problems))

        self.bm25 = BM25Index.load(self.bm25_index_path)
        if expected_articles and len(self.bm25) != expected_articles:
            raise ValueError(f"BM25 doc count mismatch: {len(self.bm25)} != {expected_articles}")

        self.pipeline = LegalAIPipeline(
            config_path=self.config_path,
            mock=False,
            bm25_index=self.bm25,
        )
        self.pipeline.config.retrieval.top_k_fusion = 60
        self.pipeline.config.retrieval.top_k_dense = max(self.pipeline.config.retrieval.top_k_dense, 350)
        self.pipeline.config.retrieval.top_k_bm25 = max(self.pipeline.config.retrieval.top_k_bm25, 350)
        self.pipeline.config.retrieval.enable_intent_retrieval = False

        self.lookup = ArticleLookup(self.corpus_path)
        self.corpus_by_article = load_corpus(self.corpus_path)
        self.bge = BgeScorer(self.config, dry_run=False)

    def route(self, question: str) -> dict[str, str]:
        routed = route_query(self.config, question)
        return {"route": routed.route, "reason": routed.reason}

    def answer_without_rag(self, qid: str | int, question: str, route: str, reason: str = "") -> dict:
        if route == SMALL_TALK_ROUTE:
            answer = answer_small_talk(self.config, question)
        else:
            answer = out_of_scope_answer()
        return {
            "id": qid,
            "question": question,
            "answer": answer,
            "relevant_docs": [],
            "relevant_articles": [],
            "_debug": {
                "route": route,
                "route_reason": reason,
                "sizes": {},
            },
        }

    def analyze(self, qid: str | int, question: str) -> dict[str, Any]:
        row = {"id": qid, "question": question}
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            rewrite_future = executor.submit(_rewrite_one, self.config, row)
            intent_future = executor.submit(_decompose_one, self.config, row)
            rewrite = rewrite_future.result()
            intents = intent_future.result()
        log.info("analysis %.2fs", time.perf_counter() - started)
        return {**rewrite, **{k: v for k, v in intents.items() if k not in {"id", "question"}}}

    def _format_global_row(
        self,
        qid: str | int,
        question: str,
        resolved_results: list[dict],
    ) -> dict:
        articles = dedupe_keep_order(
            [item.get("relevant_article_str", "") for item in resolved_results if item.get("relevant_article_str")]
        )[:60]
        return to_submission_row(qid, question, articles)

    def global_rrf_from_vectors(
        self,
        analysis: dict[str, Any],
        query_vec,
        topic_vec=None,
        bm25_hits: list[tuple[str, float]] | None = None,
        dense_query: tuple[list[tuple[str, float]], dict[str, dict]] | None = None,
        dense_topic: tuple[list[tuple[str, float]], dict[str, dict]] | None = None,
    ) -> dict:
        query = analysis.get("rewritten_query") or analysis["question"]
        rankings: list[list[tuple[str, float]]] = []
        if bm25_hits is None:
            bm25_hits = self.bm25.search(query, top_k=self.config.retrieval.top_k_bm25)
        rankings.append(bm25_hits)

        if dense_query is None:
            dense_query = self.pipeline._dense_search(query_vec)
        dense_hits, dense_payloads = dense_query
        rankings.append(dense_hits)

        if topic_vec is not None:
            if dense_topic is None:
                dense_topic = self.pipeline._dense_search(topic_vec)
            td_hits, td_payloads = dense_topic
            rankings.append(td_hits)
            dense_payloads = {**dense_payloads, **td_payloads}

        fused = rrf_fusion(rankings, k=self.config.retrieval.rrf_k)
        resolved = self.pipeline._resolve_payloads(
            fused[: self.config.retrieval.top_k_fusion],
            dense_payloads,
        )
        return self._format_global_row(analysis["id"], analysis["question"], resolved)

    def intent_rrf(self, analysis: dict[str, Any]) -> dict:
        return retrieve_ranked_hits_for_question(
            self.config,
            self.bm25,
            analysis["id"],
            analysis["question"],
            analysis.get("legal_intents", []),
        )

    def intent_rrf_from_vectors(
        self,
        analysis: dict[str, Any],
        intent_vectors: dict[int, Any],
        bm25_hits_by_intent: dict[int, list[tuple[str, float]]] | None = None,
        dense_by_intent: dict[int, tuple[list[tuple[str, float]], dict[str, dict]]] | None = None,
    ) -> dict:
        cfg = self.config.retrieval
        intents = [str(x).strip() for x in analysis.get("legal_intents", [])]
        intent_ranked_hits: list[dict] = []

        for intent_index, intent in enumerate(intents):
            if not intent:
                intent_ranked_hits.append(
                    {
                        "intent_index": intent_index,
                        "intent_id": intent_index + 1,
                        "intent": intent,
                        "ranked_articles": [],
                    }
                )
                continue

            bm25_hits = (
                bm25_hits_by_intent.get(intent_index)
                if bm25_hits_by_intent is not None
                else None
            )
            if bm25_hits is None:
                bm25_hits = self.bm25.search(intent, top_k=cfg.intent_top_k_bm25)

            dense_result = dense_by_intent.get(intent_index) if dense_by_intent is not None else None
            if dense_result is None:
                dense_result = self.pipeline._dense_search(
                    intent_vectors[intent_index],
                    limit=cfg.intent_top_k_dense,
                )
            dense_hits, dense_payloads = dense_result
            fused = rrf_fusion([bm25_hits, dense_hits], k=cfg.rrf_k)[: cfg.intent_top_k_rrf]
            resolved = self.pipeline._resolve_payloads(fused, dense_payloads)

            ranked_articles: list[dict] = []
            for rank, item in enumerate(resolved, start=1):
                article = item.get("relevant_article_str", "")
                if not article:
                    continue
                ranked_articles.append(
                    {
                        "rank": rank,
                        "chunk_id": item.get("chunk_id", ""),
                        "article": article,
                        "doc": item.get("relevant_doc_str", ""),
                        "rrf_score": float(item.get("rrf_score", 0.0)),
                    }
                )

            intent_ranked_hits.append(
                {
                    "intent_index": intent_index,
                    "intent_id": intent_index + 1,
                    "intent": intent,
                    "ranked_articles": ranked_articles,
                }
            )

        union_by_article: dict[str, dict] = {}
        for intent_row in intent_ranked_hits:
            intent_id = intent_row["intent_id"]
            intent = intent_row["intent"]
            for item in intent_row["ranked_articles"]:
                article = item["article"]
                current = union_by_article.setdefault(
                    article,
                    {
                        "chunk_id": item.get("chunk_id", ""),
                        "article": article,
                        "doc": item.get("doc", ""),
                        "matched_intents": [],
                        "best_intent_rank": item["rank"],
                        "best_intent_rrf_score": item["rrf_score"],
                        "intent_hit_count": 0,
                    },
                )
                current["matched_intents"].append(
                    {
                        "intent_id": intent_id,
                        "intent_index": intent_id - 1,
                        "intent": intent,
                        "rank": item["rank"],
                        "rrf_score": item["rrf_score"],
                    }
                )
                current["best_intent_rank"] = min(current["best_intent_rank"], item["rank"])
                current["best_intent_rrf_score"] = max(
                    current["best_intent_rrf_score"],
                    item["rrf_score"],
                )
                current["intent_hit_count"] = len(current["matched_intents"])

        union_hits = sorted(
            union_by_article.values(),
            key=lambda x: (
                int(x["best_intent_rank"]),
                -float(x["best_intent_rrf_score"]),
                x["article"],
            ),
        )

        return {
            "id": analysis["id"],
            "question": analysis["question"],
            "legal_intents": intents,
            "num_intents": len(intents),
            "intent_ranked_hits": intent_ranked_hits,
            "intent_hits_union": union_hits,
        }

    def retrieve(self, analysis: dict[str, Any]) -> tuple[dict, dict]:
        started = time.perf_counter()
        query = analysis.get("rewritten_query") or analysis["question"]
        topic = str(analysis.get("topic_description") or "").strip()
        use_topic = bool(topic and topic != query)
        intents = [str(x).strip() for x in analysis.get("legal_intents", [])]

        embed_texts = [query]
        topic_offset = None
        if use_topic:
            topic_offset = len(embed_texts)
            embed_texts.append(topic)
        intent_offsets: dict[int, int] = {}
        for intent_index, intent in enumerate(intents):
            if intent:
                intent_offsets[intent_index] = len(embed_texts)
                embed_texts.append(intent)

        max_workers = max(2, min(4, 2 + (1 if use_topic else 0) + min(len(intent_offsets), 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            global_bm25_future = executor.submit(
                self.bm25.search,
                query,
                self.config.retrieval.top_k_bm25,
            )
            intent_bm25_futures = {
                intent_index: executor.submit(
                    self.bm25.search,
                    intent,
                    self.config.retrieval.intent_top_k_bm25,
                )
                for intent_index, intent in enumerate(intents)
                if intent
            }

            vectors = self.pipeline._embedder.embed_queries(embed_texts)
            query_vec = vectors[0]
            topic_vec = vectors[topic_offset] if topic_offset is not None else None
            intent_vectors = {
                intent_index: vectors[offset]
                for intent_index, offset in intent_offsets.items()
            }

            dense_query_future = executor.submit(self.pipeline._dense_search, query_vec)
            dense_topic_future = (
                executor.submit(self.pipeline._dense_search, topic_vec)
                if topic_vec is not None
                else None
            )
            intent_dense_futures = {
                intent_index: executor.submit(
                    self.pipeline._dense_search,
                    vector,
                    self.config.retrieval.intent_top_k_dense,
                )
                for intent_index, vector in intent_vectors.items()
            }

            global_row = self.global_rrf_from_vectors(
                analysis,
                query_vec,
                topic_vec,
                bm25_hits=global_bm25_future.result(),
                dense_query=dense_query_future.result(),
                dense_topic=dense_topic_future.result() if dense_topic_future is not None else None,
            )
            intent_row = self.intent_rrf_from_vectors(
                analysis,
                intent_vectors,
                bm25_hits_by_intent={
                    intent_index: future.result()
                    for intent_index, future in intent_bm25_futures.items()
                },
                dense_by_intent={
                    intent_index: future.result()
                    for intent_index, future in intent_dense_futures.items()
                },
            )
        log.info("retrieval %.2fs", time.perf_counter() - started)
        return global_row, intent_row

    def bge_intent(self, qid: str | int, intents: list[str], keep_articles: list[str]) -> dict[int, list[dict]]:
        jobs = [(idx, intent) for idx, intent in enumerate(intents)]
        rows: dict[int, list[dict]] = {}
        if not jobs:
            return rows
        with ThreadPoolExecutor(max_workers=min(self.bge_workers, len(jobs))) as executor:
            futures = {
                executor.submit(
                    rank_one_intent,
                    self.bge,
                    qid,
                    idx,
                    intent,
                    keep_articles,
                    self.corpus_by_article,
                ): idx
                for idx, intent in jobs
            }
            for future, idx in [(future, futures[future]) for future in futures]:
                row = future.result()
                rows[idx] = row.get("ranked_articles", [])
        return rows

    def tiered_union(self, analysis: dict[str, Any], global_row: dict, intent_row: dict) -> tuple[dict, dict[int, list[dict]]]:
        qid = analysis["id"]
        intents = analysis.get("legal_intents", [])
        rrf_articles = dedupe_keep_order(global_row.get("relevant_articles", []))
        intent_hits = [
            item.get("article", "")
            for item in intent_row.get("intent_hits_union", [])
            if item.get("article")
        ]
        keep = dedupe_keep_order(rrf_articles + intent_hits)
        bge_by_intent = self.bge_intent(qid, intents, keep)
        intent_ranked_by_intent: dict[int, list[dict]] = {}
        for intent_hit in intent_row.get("intent_ranked_hits", []):
            idx = int(intent_hit.get("intent_index", 0))
            intent_ranked_by_intent[idx] = [
                {
                    "article": item.get("article", ""),
                    "doc": item.get("doc", "") or doc_ref(item.get("article", "")),
                    "rank": int(item.get("rank", 10**9)),
                    "score": float(item.get("rrf_score", 0.0)),
                }
                for item in intent_hit.get("ranked_articles", [])
                if item.get("article")
            ]
        row, diagnostics = build_entry(
            qid,
            analysis["question"],
            rrf_articles,
            bge_by_intent,
            intent_row.get("intent_hits_union", []),
            intent_ranked_by_intent,
            len(intents),
            top_b_rrf=12,
            top_n_bge=5,
            top_m_intent=5,
        )
        return row, bge_by_intent

    def stage1(self, tiered_row: dict, legal_intents: list[str]) -> dict:
        record = verify_one(
            self.config,
            self.lookup,
            tiered_row,
            legal_intents,
            strict_errors=True,
            stage1_compact_candidates=True,
            parallel_batch_rounds=True,
            batch_round_workers=self.stage1_batch_workers,
        )
        return to_submission_row(record["id"], record["question"], record["final_article_ids"])

    def penalty_cleanup(self, row: dict) -> dict:
        original = dedupe_keep_order(row.get("relevant_articles", []))
        final: list[str] = []
        for article_id in original:
            article = self.lookup.get(article_id)
            if article is None:
                raise KeyError(f"Article is missing from corpus lookup: {article_id}")
            drop, _reason = stage1_should_drop_penalty(row.get("question", ""), article)
            if not drop:
                final.append(article_id)
        if original and not final:
            final = original
        return to_submission_row(row["id"], row["question"], final)

    def final_collective(self, row: dict, legal_intents: list[str]) -> dict:
        record = {
            "id": row["id"],
            "question": row["question"],
            "final_article_ids": dedupe_keep_order(row.get("relevant_articles", [])),
        }
        out = final_collective_question(
            self.config,
            self.lookup,
            record,
            legal_intents,
            content_max_chars=2200,
            batch_size=6,
            direct_max=8,
            min_size=2,
            preserve_top1=False,
            strict_errors=True,
            system_prompt=FINAL_COLLECTIVE_SYSTEM_PROMPT,
            prompt_mode="final_precision_gemma_v5",
            compact_candidates=True,
            parallel_batch_rounds=True,
            batch_round_workers=self.final_batch_workers,
        )
        return to_submission_row(out["id"], out["question"], out["final_article_ids"])

    def enforcement_gate(self, row: dict) -> dict:
        original = dedupe_keep_order(row.get("relevant_articles", []))
        hydrated = {article_id: self.lookup.get(article_id) for article_id in original}
        missing = [article_id for article_id, article in hydrated.items() if article is None]
        if missing:
            raise KeyError(f"Articles missing from corpus lookup: {missing[:5]}")
        non_enforcement = {
            article_id
            for article_id, article in hydrated.items()
            if article is not None and not is_penalty_article(article) and not is_coercion_article(article)
        }
        dropped = 0
        final: list[str] = []
        for article_id in original:
            article = hydrated[article_id]
            assert article is not None
            reason = enforcement_drop_reason(
                row.get("question", ""),
                article,
                has_non_enforcement_alternative=bool(non_enforcement),
            )
            if reason and len(original) - dropped > 1:
                dropped += 1
                continue
            final.append(article_id)
        return to_submission_row(row["id"], row["question"], final)

    def rescue(self, final_row: dict, stage1_row: dict, intent_row: dict) -> dict:
        final_articles, _rescued = rescue_question(
            final_row.get("relevant_articles", []),
            stage1_row.get("relevant_articles", []),
            intent_row.get("intent_ranked_hits", []),
            coverage_depth=self.rescue_coverage_depth,
        )
        return to_submission_row(final_row["id"], final_row["question"], final_articles)

    def generate(self, row: dict) -> dict:
        article_refs = dedupe_keep_order(row.get("relevant_articles", []))
        articles = [self.lookup.require(article_ref) for article_ref in article_refs]
        signature = generation_signature(self.config, row["question"], article_refs, articles)
        generated = generate_one(
            self.config,
            {
                "id": row["id"],
                "question": row["question"],
                "article_refs": article_refs,
                "articles": articles,
                "known_law_ids": self.lookup.law_ids,
                "signature": signature,
            },
        )
        return {
            **row,
            "answer": generated["answer"],
        }

    def run(self, question: str, qid: str | int = "1", *, include_debug: bool = False) -> dict:
        started = time.perf_counter()
        routed = route_query(self.config, question)
        if routed.route != LEGAL_ROUTE:
            out = self.answer_without_rag(qid, question, routed.route, routed.reason)
            if include_debug:
                out["_debug"] = {
                    **out.get("_debug", {}),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            return out
        analysis = self.analyze(qid, question)
        global_row, intent_row = self.retrieve(analysis)
        tiered_row, _bge_by_intent = self.tiered_union(analysis, global_row, intent_row)
        stage1_row = self.stage1(tiered_row, analysis.get("legal_intents", []))
        cleaned_row = self.penalty_cleanup(stage1_row)
        final_row = self.final_collective(cleaned_row, analysis.get("legal_intents", []))
        gated_row = self.enforcement_gate(final_row)
        rescued_row = self.rescue(gated_row, cleaned_row, intent_row)
        answered = self.generate(rescued_row)
        elapsed = time.perf_counter() - started
        if not include_debug:
            return answered
        return {
            **answered,
            "_debug": {
                "elapsed_seconds": elapsed,
                "legal_intents": analysis.get("legal_intents", []),
                "sizes": {
                    "rrf60": len(global_row.get("relevant_articles", [])),
                    "tiered": len(tiered_row.get("relevant_articles", [])),
                    "stage1": len(stage1_row.get("relevant_articles", [])),
                    "penalty_cleanup": len(cleaned_row.get("relevant_articles", [])),
                    "final_collective": len(final_row.get("relevant_articles", [])),
                    "enforcement_gate": len(gated_row.get("relevant_articles", [])),
                    "rescue": len(rescued_row.get("relevant_articles", [])),
                },
            },
        }


def load_one_question(path: Path) -> tuple[str | int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("Single-question runner expects exactly one row")
        data = data[0]
    qid = data.get("id", "1")
    question = str(data.get("question") or "").strip()
    if not question:
        raise ValueError("Input question is empty")
    return qid, question


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run the production RAG workflow for one question in-process.")
    parser.add_argument("--config", default="config_gpu_gemini_production.yaml")
    parser.add_argument("--input", help="JSON object/list with one {id, question}")
    parser.add_argument("--question", default="")
    parser.add_argument("--id", default="1")
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--bm25-index", default="retrieval/data/bm25_index_asof_20260301.pkl")
    parser.add_argument("--expected-articles", type=int, default=82570)
    parser.add_argument("--stage1-batch-round-workers", type=int, default=2)
    parser.add_argument("--final-batch-round-workers", type=int, default=2)
    parser.add_argument("--bge-workers", type=int, default=4)
    parser.add_argument("--rescue-coverage-depth", type=int, default=4, choices=(2, 4))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.input:
        qid, question = load_one_question(Path(args.input))
    else:
        qid, question = args.id, args.question.strip()
    if not question:
        parser.error("Provide --question or --input")

    runner = SingleQuestionRagRunner(
        config_path=args.config,
        corpus_path=args.corpus,
        bm25_index_path=args.bm25_index,
        expected_articles=args.expected_articles,
        stage1_batch_workers=args.stage1_batch_round_workers,
        final_batch_workers=args.final_batch_round_workers,
        bge_workers=args.bge_workers,
        rescue_coverage_depth=args.rescue_coverage_depth,
    )
    result = runner.run(question, qid=qid, include_debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_dir:
        save_submission([result], Path(args.output_dir))


if __name__ == "__main__":
    main()
