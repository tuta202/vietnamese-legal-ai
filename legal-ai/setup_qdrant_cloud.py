"""
Validate and bootstrap the Vertex AI / Qdrant stack.

The indexing path is designed for a final corpus build:
- resumable at point level by checking existing Qdrant ids before embedding
- parallel indexing with a small worker count
- retry for Gemini embedding in VertexEmbedder plus explicit Qdrant upsert retry
- wait=True upserts
- final exact count and full expected-id verification
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.config import RetrievalConfig, load_config, validate_config
from vertex_backends import VertexEmbedder, VertexQueryRewriter

_LOG_LOCK = threading.Lock()
_PROGRESS_LOG = _ROOT / "outputs" / "embed_clean_v1_progress.jsonl"
_ERROR_LOG = _ROOT / "outputs" / "embed_clean_v1_errors.jsonl"
_MISSING_IDS_LOG = _ROOT / "outputs" / "embed_clean_v1_missing_ids.json"


def _qdrant_client(config: RetrievalConfig):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=config.qdrant.url,
        api_key=config.qdrant.api_key,
        timeout=120,
    )


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _payload(article: dict) -> dict:
    return {
        "chunk_id": article["chunk_id"],
        "law_id": article["law_id"],
        "law_type": article["law_type"],
        "law_name": article["law_name"],
        "dieu_number": article["dieu_number"],
        "dieu_title": article["dieu_title"],
        "content": article["content"],
        "relevant_doc_str": article["relevant_doc_str"],
        "relevant_article_str": article["relevant_article_str"],
    }


def _ensure_collection(client, config: RetrievalConfig, force: bool) -> None:
    from qdrant_client.models import Distance, VectorParams

    name = config.qdrant.collection
    distance = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
    }.get(str(config.qdrant.distance).lower(), Distance.COSINE)

    existing = {collection.name for collection in client.get_collections().collections}
    if force and name in existing:
        client.delete_collection(name)
        existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config.qdrant.vector_size,
                distance=distance,
            ),
        )
        print(f"  Created collection '{name}' ({config.qdrant.vector_size}d, {distance})")


def _batch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def _index_one_batch(config: RetrievalConfig, articles: list[dict], start: int, end: int) -> dict:
    from qdrant_client.models import PointStruct

    client = _qdrant_client(config)
    embedder = VertexEmbedder(config, dry_run=False)
    collection = config.qdrant.collection
    batch = articles[start:end]
    ids = [str(uuid.UUID(article["chunk_id"])) for article in batch]

    try:
        present = client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        present_ids = {str(point.id) for point in present}
    except Exception as exc:
        raise RuntimeError(f"retrieve failed for batch {start}:{end}: {exc}") from exc

    todo = [(article, point_id) for article, point_id in zip(batch, ids) if point_id not in present_ids]
    embedded = 0
    if todo:
        texts = [embedder._format_document(article) for article, _ in todo]
        vectors = embedder.embed_documents(texts)
        embedded = len(todo)
        points = [
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=_payload(article),
            )
            for (article, point_id), vector in zip(todo, vectors)
        ]

        for attempt in range(1, 6):
            try:
                client.upsert(collection_name=collection, points=points, wait=True)
                break
            except Exception as exc:
                if attempt == 5:
                    raise RuntimeError(f"upsert failed for batch {start}:{end}: {exc}") from exc
                time.sleep(2 * attempt)

    row = {
        "batch_start": start,
        "batch_end": end,
        "batch_size": len(batch),
        "already_present": len(present_ids),
        "embedded": embedded,
        "status": "ok",
    }
    _append_jsonl(_PROGRESS_LOG, row)
    return row


def _index_corpus_parallel(config: RetrievalConfig, articles: list[dict], workers: int) -> int:
    total = len(articles)
    batch_size = max(1, int(config.embedding.batch_size))
    ranges = _batch_ranges(total, batch_size)
    workers = max(1, int(workers))

    _PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    _PROGRESS_LOG.unlink(missing_ok=True)
    _ERROR_LOG.unlink(missing_ok=True)
    _MISSING_IDS_LOG.unlink(missing_ok=True)

    print(f"  batches={len(ranges)} batch_size={batch_size} workers={workers}")
    completed = 0
    embedded = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_index_one_batch, config, articles, start, end): (start, end)
            for start, end in ranges
        }
        for future in as_completed(futures):
            start, end = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                error = {
                    "batch_start": start,
                    "batch_end": end,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _append_jsonl(_ERROR_LOG, error)
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"batch {start}:{end} failed after retries; see {_ERROR_LOG}"
                ) from exc

            completed += int(row["batch_size"])
            embedded += int(row["embedded"])
            elapsed = max(1e-9, time.time() - t0)
            rate = completed / elapsed
            eta = (total - completed) / rate if rate else 0
            print(
                f"  {completed}/{total} confirmed "
                f"(embedded this run: {embedded}, {rate:.1f} art/s, ETA {eta / 60:.0f} min)",
                end="\r",
                flush=True,
            )

    print()
    print(f"  embedded {embedded} new points this run; {total - embedded} were already present")
    return total


def _verify_complete_collection(client, config: RetrievalConfig, articles: list[dict]) -> bool:
    collection = config.qdrant.collection
    expected = len(articles)
    count = client.count(collection_name=collection, exact=True).count
    print(f"  exact point count: {count} (expected {expected})")

    expected_ids = [str(uuid.UUID(article["chunk_id"])) for article in articles]
    missing: list[str] = []
    chunk_size = 256
    for start in range(0, len(expected_ids), chunk_size):
        ids = expected_ids[start:start + chunk_size]
        present = client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        present_ids = {str(point.id) for point in present}
        missing.extend([point_id for point_id in ids if point_id not in present_ids])

    if missing:
        _MISSING_IDS_LOG.write_text(
            json.dumps({"collection": collection, "missing_ids": missing}, indent=2),
            encoding="utf-8",
        )
        print(f"  FAIL missing ids: {len(missing)} (see {_MISSING_IDS_LOG})")
        return False

    if count != expected:
        print("  FAIL count mismatch even though all expected ids are present")
        return False

    print("  OK all expected chunk ids are present")
    return True


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Setup/test Vertex + Qdrant Cloud")
    parser.add_argument("--config", default="config_vertex.yaml")
    parser.add_argument("--corpus", default="corpus/data/corpus.json")
    parser.add_argument("--force", action="store_true", help="Recreate the collection")
    parser.add_argument("--test-only", action="store_true", help="Only connectivity tests; skip corpus indexing")
    parser.add_argument("--workers", type=int, default=1, help="Parallel embed/upsert workers")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  Vertex AI + Qdrant Cloud setup")
    print("=" * 60)

    print("\n[1/7] Validating config")
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"  FAIL {error}")
        sys.exit(1)
    print("  OK config")

    embedder = VertexEmbedder(config, dry_run=False)
    rewriter = VertexQueryRewriter(config, mock=False)

    print("\n[2/7] Testing embedding")
    vec = embedder.embed_query("Doanh nghiep nho va vua la gi?")
    if len(vec) != config.embedding.dimension:
        raise SystemExit(f"dimension mismatch: got {len(vec)}, expected {config.embedding.dimension}")
    print(f"  OK embedding dim = {len(vec)}")

    print("\n[3/7] Testing Qdrant connection")
    client = _qdrant_client(config)
    collections = client.get_collections().collections
    print(f"  OK connected - {len(collections)} existing collection(s)")

    print("\n[4/7] Testing LLM query rewrite")
    rewritten = rewriter.rewrite("Nguoi lao dong duoc nghi bao nhieu ngay phep nam?")
    print(f"  OK rewritten_query: {rewritten['rewritten_query'][:80]}")

    if args.test_only:
        print("\n[5/7] --test-only set; skipping corpus indexing")
        print("\nDone (test-only).")
        return

    print(f"\n[5/7] Indexing corpus from {args.corpus}")
    _ensure_collection(client, config, force=args.force)
    articles = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["articles"]
    print(f"  {len(articles)} articles loaded")
    indexed = _index_corpus_parallel(config, articles, workers=args.workers)
    print(f"  OK indexed {indexed} articles into '{config.qdrant.collection}'")

    print("\n[6/7] Verifying full collection")
    if not _verify_complete_collection(client, config, articles):
        sys.exit(2)

    print("\n[7/7] Test search: 'doanh nghiep nho va vua' top 5")
    q_vec = embedder.embed_query("doanh nghiep nho va vua").tolist()
    response = client.query_points(
        collection_name=config.qdrant.collection,
        query=q_vec,
        limit=5,
        with_payload=True,
    )
    for rank, hit in enumerate(response.points, 1):
        payload = hit.payload or {}
        print(
            f"  {rank}. [{payload.get('law_id', '')}] {payload.get('dieu_number', '')} "
            f"- {str(payload.get('dieu_title', ''))[:60]} (score={hit.score:.4f})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
