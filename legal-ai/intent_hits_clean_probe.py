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

from pipeline import PipelineState  # noqa: E402
from retrieval.config import load_config, validate_config  # noqa: E402
from retrieval.hybrid_search import rrf_fusion  # noqa: E402
from retrieval.intent_decomposer import LegalIntentDecomposer  # noqa: E402
from retrieval.bm25_index import BM25Index  # noqa: E402
from submit import save_submission  # noqa: E402
from vertex_backends import VertexEmbedder, VertexQueryRewriter  # noqa: E402


log = logging.getLogger("intent_hits_clean_probe")
_THREAD_LOCAL = threading.local()


def load_clean_bm25(index_path: Path, expected_count: int) -> BM25Index:
    log.info("Loading clean BM25 from %s", index_path)
    index = BM25Index.load(index_path)
    if len(index) != expected_count:
        raise ValueError(f"BM25 doc count mismatch: {len(index)} != {expected_count}")
    return index


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    records: dict = {}
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
            if qid is not None and row.get("relevant_articles"):
                records[qid] = row
    return records


class WorkerContext:
    def __init__(self, config, bm25: BM25Index) -> None:
        from qdrant_client import QdrantClient  # noqa: PLC0415

        self.config = config
        self.bm25 = bm25
        self.embedder = VertexEmbedder(config, dry_run=False)
        self.rewriter = VertexQueryRewriter(config, mock=False)
        self.decomposer = LegalIntentDecomposer(
            config,
            mock=False,
            chat_complete=self.rewriter._chat_complete,
        )
        q = config.qdrant
        self.qdrant = QdrantClient(url=q.url, api_key=q.api_key) if q.url else QdrantClient(q.host, port=q.port)

    def dense_search(self, vector, limit: int) -> tuple[list[tuple[str, float]], dict[str, dict]]:
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        resp = self.qdrant.query_points(
            collection_name=self.config.qdrant.collection,
            query=vec_list,
            limit=limit,
            with_payload=True,
        )
        points = getattr(resp, "points", resp)
        hits: list[tuple[str, float]] = []
        payloads: dict[str, dict] = {}
        for point in points:
            payload = getattr(point, "payload", None)
            if payload is None and isinstance(point, dict):
                payload = point.get("payload")
            payload = payload or {}
            rid = getattr(point, "id", None)
            if rid is None and isinstance(point, dict):
                rid = point.get("id")
            chunk_id = payload.get("chunk_id") or str(rid)
            score = getattr(point, "score", None)
            if score is None and isinstance(point, dict):
                score = point.get("score", 0.0)
            hits.append((chunk_id, float(score or 0.0)))
            if payload:
                payloads[chunk_id] = payload
        return hits, payloads

    def resolve_payloads(
        self,
        fused: list[tuple[str, float]],
        dense_payloads: dict[str, dict],
    ) -> list[dict]:
        results: list[dict] = []
        for chunk_id, score in fused:
            payload = self.bm25.get_payload(chunk_id)
            if payload is None:
                qp = dense_payloads.get(chunk_id) or {}
                law_id = qp.get("law_id", "")
                law_type = qp.get("law_type", "")
                law_name = qp.get("law_name", "")
                dieu_number = qp.get("dieu_number") or qp.get("article_number", "")
                payload = {
                    "chunk_id": qp.get("chunk_id", chunk_id),
                    "law_id": law_id,
                    "law_type": law_type,
                    "law_name": law_name,
                    "dieu_number": dieu_number,
                    "dieu_title": qp.get("dieu_title") or qp.get("article_title", ""),
                    "content": qp.get("content", ""),
                    "relevant_doc_str": qp.get("relevant_doc_str") or f"{law_id}|{law_type} {law_name}",
                    "relevant_article_str": qp.get("relevant_article_str") or f"{law_id}|{law_type} {law_name}|{dieu_number}",
                }
            results.append({**payload, "rrf_score": score})
        return results

    def retrieve_intent_hits(self, intents: list[str]) -> list[dict]:
        cfg = self.config.retrieval
        by_chunk: dict[str, dict] = {}
        for intent_id, intent in enumerate(intents, start=1):
            text = intent.strip()
            if not text:
                continue
            bm25_hits = self.bm25.search(text, top_k=cfg.intent_top_k_bm25)
            vec = self.embedder.embed_query(text)
            dense_hits, dense_payloads = self.dense_search(vec, cfg.intent_top_k_dense)
            fused = rrf_fusion([bm25_hits, dense_hits], k=cfg.rrf_k)[: cfg.intent_top_k_rrf]
            resolved = self.resolve_payloads(fused, dense_payloads)
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


def get_worker_context(config, bm25: BM25Index) -> WorkerContext:
    ctx = getattr(_THREAD_LOCAL, "ctx", None)
    if ctx is None:
        ctx = WorkerContext(config, bm25)
        _THREAD_LOCAL.ctx = ctx
    return ctx


def retrieve_intent_hits_only(
    config,
    bm25: BM25Index,
    qid: int | str,
    question: str,
) -> dict:
    try:
        ctx = get_worker_context(config, bm25)
        intents = ctx.decomposer.decompose(question).intents
        intent_hits = ctx.retrieve_intent_hits(intents)

        submission_entry = format_submission_entry(qid, question, intent_hits)

        return {
            **submission_entry,
            "legal_intents": intents,
            "num_intents": len(intents),
            "intent_hit_count": len(intent_hits),
            "intent_hits": [
                {
                    "chunk_id": item.get("chunk_id", ""),
                    "article": item.get("relevant_article_str", ""),
                    "doc": item.get("relevant_doc_str", ""),
                    "intent_ids": item.get("intent_ids", []),
                    "intent_rank": item.get("intent_rank"),
                    "intent_rrf_score": item.get("intent_rrf_score"),
                    "rrf_score": item.get("rrf_score"),
                }
                for item in intent_hits
            ],
        }
    except Exception as exc:
        log.exception("Q%s failed; emitting empty fallback", qid)
        return {
            "id": qid,
            "question": question,
            "answer": "",
            "relevant_docs": [],
            "relevant_articles": [],
            "legal_intents": [],
            "num_intents": 0,
            "intent_hit_count": 0,
            "intent_hits": [],
            "error": str(exc),
        }


def format_submission_entry(qid: int | str, question: str, articles: list[dict]) -> dict:
    seen_docs: set[str] = set()
    docs: list[str] = []
    for article in articles:
        doc = article.get("relevant_doc_str", "")
        if doc and doc not in seen_docs:
            seen_docs.add(doc)
            docs.append(doc)

    seen_articles: set[str] = set()
    rel_articles: list[str] = []
    for article in articles:
        rel = article.get("relevant_article_str", "")
        if rel and rel not in seen_articles:
            seen_articles.add(rel)
            rel_articles.append(rel)

    return {
        "id": qid,
        "question": question,
        "answer": "",
        "relevant_docs": docs,
        "relevant_articles": rel_articles,
    }


def to_submission(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row.get("id"),
            "question": row.get("question", ""),
            "answer": "",
            "relevant_docs": row.get("relevant_docs", []),
            "relevant_articles": row.get("relevant_articles", []),
        }
        for row in rows
    ]


def write_ordered_outputs(
    output_path: Path,
    submission_dir: Path,
    questions: list[dict],
    records: dict,
) -> list[dict]:
    ordered = [
        records.get(
            q["id"],
            {
                "id": q["id"],
                "question": q["question"],
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": [],
                "legal_intents": [],
                "num_intents": 0,
                "intent_hit_count": 0,
                "intent_hits": [],
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
    save_submission(to_submission(ordered), submission_dir)
    return ordered


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run legal intent decomposition + intent-only retrieval on the clean corpus."
    )
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--cache", default="outputs/intent_hits_clean_cache.jsonl")
    parser.add_argument("--output", default="outputs/intent_hits_clean_results.json")
    parser.add_argument("--submission-dir", default="outputs/submission_intent_hits_clean")
    parser.add_argument("--bm25-index", default="retrieval/data/bm25_index_clean_v1.pkl")
    parser.add_argument("--expected-count", type=int, default=86492)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")
    config.retrieval.enable_intent_retrieval = True
    config.retrieval.intent_top_k_bm25 = 50
    config.retrieval.intent_top_k_dense = 50
    config.retrieval.intent_top_k_rrf = 10
    bm25 = load_clean_bm25(Path(args.bm25_index), args.expected_count)

    cache_path = Path(args.cache)
    records = load_cache(cache_path) if args.resume else {}
    log.info("Resume loaded %d/%d completed rows", len(records), len(questions))

    todo = [q for q in questions if q["id"] not in records]
    log.info(
        "Running intent-only retrieval: total=%d todo=%d workers=%d collection=%s",
        len(questions),
        len(todo),
        args.workers,
        config.qdrant.collection,
    )
    log.info(
        "Intent source depths: bm25=%d dense=%d rrf_each_intent=%d",
        config.retrieval.intent_top_k_bm25,
        config.retrieval.intent_top_k_dense,
        config.retrieval.intent_top_k_rrf,
    )

    lock = threading.Lock()
    started = time.time()
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(retrieve_intent_hits_only, config, bm25, q["id"], q["question"]): q["id"]
            for q in todo
        }
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                records[row["id"]] = row
                processed += 1
            if row.get("relevant_articles"):
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

    ordered = write_ordered_outputs(
        Path(args.output),
        Path(args.submission_dir),
        questions,
        records,
    )
    sizes = [len(r.get("relevant_articles", [])) for r in ordered]
    intents = [len(r.get("legal_intents", [])) for r in ordered]
    log.info(
        "Final rows=%d empty=%d intent_hits/q min/max/mean=%d/%d/%.2f intents/q min/max/mean=%d/%d/%.2f",
        len(ordered),
        sum(1 for s in sizes if s == 0),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        sum(sizes) / len(sizes) if sizes else 0,
        min(intents) if intents else 0,
        max(intents) if intents else 0,
        sum(intents) / len(intents) if intents else 0,
    )


if __name__ == "__main__":
    main()
