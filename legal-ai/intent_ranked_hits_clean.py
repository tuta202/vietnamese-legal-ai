from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_hits_clean_probe import WorkerContext, load_clean_bm25  # noqa: E402
from retrieval.config import load_config, validate_config  # noqa: E402
from retrieval.hybrid_search import rrf_fusion  # noqa: E402


log = logging.getLogger("intent_ranked_hits_clean")
_THREAD_LOCAL = threading.local()


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def load_cache(path: Path) -> dict[object, dict]:
    if not path.exists():
        return {}
    records: dict[object, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("id")
            if qid is not None and row.get("intent_ranked_hits"):
                records[qid] = row
    return records


def load_existing_intents(path: Path) -> dict[object, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["id"]: [str(x).strip() for x in row.get("legal_intents", []) if str(x).strip()]
        for row in rows
    }


def get_worker_context(config, bm25) -> WorkerContext:
    ctx = getattr(_THREAD_LOCAL, "ctx", None)
    if ctx is None:
        ctx = WorkerContext(config, bm25)
        _THREAD_LOCAL.ctx = ctx
    return ctx


def retrieve_ranked_hits_for_question(config, bm25, qid, question: str, intents: list[str]) -> dict:
    ctx = get_worker_context(config, bm25)
    cfg = config.retrieval
    intent_ranked_hits: list[dict] = []

    for intent_index, intent in enumerate(intents):
        text = intent.strip()
        if not text:
            intent_ranked_hits.append(
                {
                    "intent_index": intent_index,
                    "intent_id": intent_index + 1,
                    "intent": intent,
                    "ranked_articles": [],
                }
            )
            continue

        bm25_hits = ctx.bm25.search(text, top_k=cfg.intent_top_k_bm25)
        vec = ctx.embedder.embed_query(text)
        dense_hits, dense_payloads = ctx.dense_search(vec, cfg.intent_top_k_dense)
        fused = rrf_fusion([bm25_hits, dense_hits], k=cfg.rrf_k)[: cfg.intent_top_k_rrf]
        resolved = ctx.resolve_payloads(fused, dense_payloads)

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
                "intent": text,
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
            current["best_intent_rrf_score"] = max(current["best_intent_rrf_score"], item["rrf_score"])
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
        "id": qid,
        "question": question,
        "legal_intents": intents,
        "num_intents": len(intents),
        "intent_ranked_hits": intent_ranked_hits,
        "intent_hits_union": union_hits,
    }


def write_ordered_output(output_path: Path, questions: list[dict], records: dict) -> list[dict]:
    ordered = [
        records.get(
            q["id"],
            {
                "id": q["id"],
                "question": q["question"],
                "legal_intents": [],
                "num_intents": 0,
                "intent_ranked_hits": [],
                "intent_hits_union": [],
            },
        )
        for q in questions
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            tmp.replace(output_path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))
    return ordered


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Rerun intent retrieval from existing legal_intents and store raw per-intent ranked hits."
    )
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--intents-source", default="outputs/intent_hits_clean_results.json")
    parser.add_argument("--cache", default="outputs/intent_ranked_hits_clean_cache.jsonl")
    parser.add_argument("--output", default="outputs/intent_ranked_hits_clean_results.json")
    parser.add_argument("--bm25-index", default="retrieval/data/bm25_index_clean_v1.pkl")
    parser.add_argument("--expected-count", type=int, default=86492)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    intents_by_qid = load_existing_intents(Path(args.intents_source))

    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")
    config.retrieval.intent_top_k_bm25 = 50
    config.retrieval.intent_top_k_dense = 50
    config.retrieval.intent_top_k_rrf = 10
    bm25 = load_clean_bm25(Path(args.bm25_index), args.expected_count)

    cache_path = Path(args.cache)
    records = load_cache(cache_path) if args.resume else {}
    todo = [q for q in questions if q["id"] not in records]
    log.info(
        "Running raw per-intent retrieval: total=%d todo=%d workers=%d collection=%s",
        len(questions),
        len(todo),
        args.workers,
        config.qdrant.collection,
    )

    lock = threading.Lock()
    processed = 0
    started = time.time()

    def work(q: dict) -> dict:
        intents = intents_by_qid.get(q["id"], [])
        return retrieve_ranked_hits_for_question(config, bm25, q["id"], q["question"], intents)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, q): q["id"] for q in todo}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                log.warning("Q%s failed: %s", qid, exc)
                processed += 1
                continue
            with lock:
                records[row["id"]] = row
                processed += 1
            append_jsonl(cache_path, row, lock)
            if processed % 10 == 0 or processed == len(todo):
                rate = processed / max(time.time() - started, 1e-6)
                eta = (len(todo) - processed) / rate if rate else 0
                log.info(
                    "%d/%d done (%.2f q/s, ETA %.1f min)",
                    len(records),
                    len(questions),
                    rate,
                    eta / 60,
                )

    ordered = write_ordered_output(Path(args.output), questions, records)
    sizes = [len(r.get("intent_hits_union", [])) for r in ordered]
    raw_sizes = [
        sum(len(x.get("ranked_articles", [])) for x in r.get("intent_ranked_hits", []))
        for r in ordered
    ]
    log.info(
        "Final rows=%d empty=%d union min/max/mean=%d/%d/%.2f raw min/max/mean=%d/%d/%.2f",
        len(ordered),
        sum(1 for s in sizes if s == 0),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        sum(sizes) / len(sizes) if sizes else 0,
        min(raw_sizes) if raw_sizes else 0,
        max(raw_sizes) if raw_sizes else 0,
        sum(raw_sizes) / len(raw_sizes) if raw_sizes else 0,
    )


if __name__ == "__main__":
    main()
