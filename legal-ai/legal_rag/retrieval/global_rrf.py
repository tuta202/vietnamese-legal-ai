from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.pipeline import LegalAIPipeline, PipelineState  # noqa: E402
from legal_rag.retrieval.bm25_index import BM25Index  # noqa: E402
from legal_rag.output.submission import save_submission  # noqa: E402


log = logging.getLogger("legal_rag.retrieval.global_rrf")


def load_clean_bm25(index_path: Path, expected_count: int) -> BM25Index:
    log.info("Loading clean BM25 from %s", index_path)
    index = BM25Index.load(index_path)
    if len(index) != expected_count:
        raise ValueError(f"BM25 doc count mismatch: {len(index)} != {expected_count}")
    return index


def retrieve_rrf_only(
    pipeline: LegalAIPipeline,
    qid: int | str,
    question: str,
    rewritten_query: str | None = None,
    topic_description: str | None = None,
    strict_errors: bool = False,
) -> dict:
    try:
        state = PipelineState(question_id=qid, question=question)
        if rewritten_query:
            state.rewritten_query = rewritten_query
            state.topic_description = topic_description or ""
        else:
            state = pipeline.step_rewrite(state)
        state = pipeline.step_retrieve(state)
        state.reranked_results = state.fused_results
        state.answer = ""
        state = pipeline.step_format(state)
        return state.submission_entry
    except Exception:
        if strict_errors:
            raise
        log.exception("Q%s failed; emitting empty fallback", qid)
        return {
            "id": qid,
            "question": question,
            "answer": "",
            "relevant_docs": [],
            "relevant_articles": [],
        }


def persist_ordered(out: Path, questions: list[dict], results: dict) -> None:
    ordered = [results[q["id"]] for q in questions if q["id"] in results]
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def load_questions(input_path: Path, cached_analysis: Path | None) -> list[dict]:
    if cached_analysis and cached_analysis.exists():
        cached = json.loads(cached_analysis.read_text(encoding="utf-8"))
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "rewritten_query": row.get("rewritten_query") or "",
                "topic_description": row.get("topic_description") or "",
            }
            for row in cached
        ]
    return json.loads(input_path.read_text(encoding="utf-8"))


def derive_prefix_submission(rows: list[dict], top_k: int) -> list[dict]:
    derived: list[dict] = []
    for row in rows:
        articles = list(row.get("relevant_articles", []))[:top_k]
        seen_docs: set[str] = set()
        docs: list[str] = []
        for article in articles:
            parts = str(article).split("|")
            if len(parts) >= 2:
                doc = "|".join(parts[:2])
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    docs.append(doc)
        derived.append(
            {
                "id": row.get("id"),
                "question": row.get("question", ""),
                "answer": "",
                "relevant_docs": docs,
                "relevant_articles": articles,
            }
        )
    return derived


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run global RRF retrieval over the clean corpus and export topK submissions."
    )
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--cached-analysis", default="")
    parser.add_argument("--output", default="outputs/rrf_top300_clean_results.json")
    parser.add_argument("--submission-prefix", default="outputs/submission_rrf_top")
    parser.add_argument("--bm25-index", default="retrieval/data/bm25_index_asof_20260301.pkl")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument(
        "--source-top-k",
        type=int,
        default=None,
        help="Per-source BM25/dense depth before RRF. Defaults to --top-k.",
    )
    parser.add_argument("--export-ks", default="100,200,300")
    parser.add_argument("--expected-count", type=int, default=82570)
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help="Leave failed questions incomplete for resume instead of emitting empty fallbacks.",
    )
    args = parser.parse_args()

    cached_analysis = Path(args.cached_analysis) if args.cached_analysis else None
    questions = load_questions(Path(args.input), cached_analysis)
    if args.limit:
        questions = questions[: args.limit]

    clean_bm25 = load_clean_bm25(Path(args.bm25_index), args.expected_count)
    pipeline = LegalAIPipeline(config_path=args.config, mock=False, bm25_index=clean_bm25)
    source_top_k = args.source_top_k or args.top_k
    pipeline.config.retrieval.top_k_fusion = args.top_k
    pipeline.config.retrieval.top_k_dense = max(pipeline.config.retrieval.top_k_dense, source_top_k)
    pipeline.config.retrieval.top_k_bm25 = max(pipeline.config.retrieval.top_k_bm25, source_top_k)
    pipeline.config.retrieval.enable_intent_retrieval = False

    out = Path(args.output)
    results: dict = {}
    if args.resume and out.exists():
        for row in json.loads(out.read_text(encoding="utf-8")):
            if row.get("id") is not None and row.get("relevant_articles"):
                results[row["id"]] = row
        log.info("Resume loaded %d/%d completed rows", len(results), len(questions))

    todo = [q for q in questions if q["id"] not in results]
    log.info(
        "Running clean RRF top%d: total=%d todo=%d workers=%d collection=%s",
        args.top_k,
        len(questions),
        len(todo),
        args.workers,
        pipeline.config.qdrant.collection,
    )
    log.info(
        "RRF source depths: bm25=%d dense=%d fusion=%d",
        pipeline.config.retrieval.top_k_bm25,
        pipeline.config.retrieval.top_k_dense,
        pipeline.config.retrieval.top_k_fusion,
    )
    lock = threading.Lock()
    started = time.time()
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                retrieve_rrf_only,
                pipeline,
                q["id"],
                q["question"],
                q.get("rewritten_query"),
                q.get("topic_description"),
                args.strict_errors,
            ): q["id"]
            for q in todo
        }
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                log.error("Q%s failed and remains incomplete: %s", qid, exc)
                processed += 1
                continue
            with lock:
                results[row["id"]] = row
                processed += 1
                if processed % 10 == 0 or processed == len(todo):
                    rate = processed / max(time.time() - started, 1e-6)
                    eta = (len(todo) - processed) / rate if rate else 0
                    log.info(
                        "%d/%d done (%.2f q/s, ETA %.1f min)",
                        len(results),
                        len(questions),
                        rate,
                        eta / 60,
                    )
                    persist_ordered(out, questions, results)

    missing = [q["id"] for q in questions if q["id"] not in results]
    if missing and args.strict_errors:
        persist_ordered(out, questions, results)
        raise RuntimeError(f"{len(missing)} global retrieval rows missing; resume required: {missing[:20]}")

    for q in questions:
        results.setdefault(
            q["id"],
            {
                "id": q["id"],
                "question": q["question"],
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": [],
            },
        )

    persist_ordered(out, questions, results)
    ordered = [results[q["id"]] for q in questions]
    sizes = [len(r.get("relevant_articles", [])) for r in ordered]
    log.info(
        "Final top%d rows=%d empty=%d articles/q min/max/mean=%d/%d/%.2f",
        args.top_k,
        len(ordered),
        sum(1 for s in sizes if s == 0),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        sum(sizes) / len(sizes) if sizes else 0,
    )

    export_ks = sorted({int(k.strip()) for k in args.export_ks.split(",") if k.strip()})
    for top_k in export_ks:
        rows = derive_prefix_submission(ordered, top_k)
        save_submission(rows, f"{args.submission_prefix}{top_k}_clean")


if __name__ == "__main__":
    main()
