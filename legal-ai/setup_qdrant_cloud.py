"""
Validate and bootstrap the configured embedding / Qdrant stack.

The indexing path is designed for a final corpus build:
- resumable at point level by checking existing Qdrant ids before embedding
- parallel indexing with a small worker count
- retry in the selected embedder plus explicit Qdrant upsert retry
- wait=True upserts
- final exact count and full expected-id verification
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.config import RetrievalConfig, load_config, validate_config
from retrieval.embedder import embedding_text_sha256

_LOG_LOCK = threading.Lock()
_PROGRESS_LOG = _ROOT / "outputs" / "embed_asof_20260301_v1_progress.jsonl"
_ERROR_LOG = _ROOT / "outputs" / "embed_asof_20260301_v1_errors.jsonl"
_MISSING_IDS_LOG = _ROOT / "outputs" / "embed_asof_20260301_v1_missing_ids.json"


def _acquire_collection_lock(collection: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in collection)
    lock_path = _ROOT / "outputs" / f".{safe_name}.embedding.running"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            text = lock_path.read_text(encoding="utf-8")
            pid = int(next(line for line in text.splitlines() if line.startswith("pid=")).split("=", 1)[1])
            os.kill(pid, 0)
        except (OSError, ValueError, StopIteration):
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Collection build is already active with PID {pid}: {lock_path}")
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(f"pid={os.getpid()}\ncollection={collection}\n")
    atexit.register(lambda: lock_path.unlink(missing_ok=True))
    return lock_path


def _qdrant_client(config: RetrievalConfig):
    from qdrant_client import QdrantClient

    if config.qdrant.url:
        return QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key or None,
            timeout=120,
        )
    return QdrantClient(host=config.qdrant.host, port=config.qdrant.port, timeout=120)


def _make_embedding_components(config: RetrievalConfig):
    if config.backend == "vertex_ai":
        from vertex_backends import VertexEmbedder, VertexQueryRewriter

        return VertexEmbedder(config, dry_run=False), VertexQueryRewriter(config, mock=False)
    if config.backend == "gpu":
        from gpu_backends import make_gpu_components

        embedder, rewriter, _, _ = make_gpu_components(config, mock=False)
        return embedder, rewriter
    raise ValueError(f"Unsupported backend: {config.backend}")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _embedding_model_name(config: RetrievalConfig) -> str:
    return config.gpu.embed_model if config.backend == "gpu" else config.models.embedder


def _payload(article: dict, config: RetrievalConfig) -> dict:
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
        "embedding_backend": config.backend,
        "embedding_model": _embedding_model_name(config),
        "embedding_text_sha256": embedding_text_sha256(article),
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
    if name in existing:
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError(f"Collection '{name}' uses named vectors; expected one unnamed vector")
        actual_size = int(vectors.size)
        actual_distance = str(vectors.distance).split(".")[-1].lower()
        expected_distance = str(distance).split(".")[-1].lower()
        if actual_size != config.qdrant.vector_size or actual_distance != expected_distance:
            raise ValueError(
                f"Collection '{name}' schema mismatch: size={actual_size}, "
                f"distance={actual_distance}; expected size={config.qdrant.vector_size}, "
                f"distance={expected_distance}. Use a new collection name or --force."
            )
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


def _validate_vectors(vectors, expected_count: int, expected_dim: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    expected_shape = (expected_count, expected_dim)
    if array.shape != expected_shape:
        raise ValueError(f"embedding shape mismatch: got {array.shape}, expected {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError("embedding output contains NaN or Inf")
    return array


def _retrieve_points(client, collection: str, ids: list[str], start: int, end: int):
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            return client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2 * attempt)
    raise RuntimeError(f"retrieve failed for batch {start}:{end}: {last_error}") from last_error


def _index_one_batch(config: RetrievalConfig, articles: list[dict], start: int, end: int) -> dict:
    from qdrant_client.models import PointStruct

    client = _qdrant_client(config)
    embedder, _ = _make_embedding_components(config)
    collection = config.qdrant.collection
    batch = articles[start:end]
    ids = [str(uuid.UUID(article["chunk_id"])) for article in batch]

    expected_hashes = {
        point_id: embedding_text_sha256(article)
        for article, point_id in zip(batch, ids)
    }
    expected_model = _embedding_model_name(config)
    present = _retrieve_points(client, collection, ids, start, end)
    present_ids = {
        str(point.id)
        for point in present
        if (point.payload or {}).get("embedding_text_sha256") == expected_hashes.get(str(point.id))
        and (point.payload or {}).get("embedding_backend") == config.backend
        and (point.payload or {}).get("embedding_model") == expected_model
    }

    todo = [(article, point_id) for article, point_id in zip(batch, ids) if point_id not in present_ids]
    embedded = 0
    if todo:
        texts = [embedder._format_document(article) for article, _ in todo]
        vectors = _validate_vectors(
            embedder.embed_documents(texts),
            expected_count=len(todo),
            expected_dim=config.embedding.dimension,
        )
        embedded = len(todo)
        points = [
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=_payload(article, config),
            )
            for (article, point_id), vector in zip(todo, vectors)
        ]

        todo_ids = {point_id for _, point_id in todo}
        for attempt in range(1, 6):
            try:
                client.upsert(collection_name=collection, points=points, wait=True)
                confirmed = _retrieve_points(client, collection, list(todo_ids), start, end)
                confirmed_ids = {
                    str(point.id)
                    for point in confirmed
                    if (point.payload or {}).get("embedding_text_sha256") == expected_hashes.get(str(point.id))
                    and (point.payload or {}).get("embedding_backend") == config.backend
                    and (point.payload or {}).get("embedding_model") == expected_model
                }
                missing_after_upsert = todo_ids - confirmed_ids
                if missing_after_upsert:
                    raise RuntimeError(
                        f"{len(missing_after_upsert)} points missing after wait=True upsert"
                    )
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
    last_count_error: Exception | None = None
    count = -1
    for attempt in range(1, 6):
        try:
            count = client.count(collection_name=collection, exact=True).count
            break
        except Exception as exc:
            last_count_error = exc
            if attempt < 5:
                time.sleep(2 * attempt)
    if count < 0:
        raise RuntimeError(f"exact count failed after retries: {last_count_error}") from last_count_error
    print(f"  exact point count: {count} (expected {expected})")

    expected_ids = [str(uuid.UUID(article["chunk_id"])) for article in articles]
    missing: list[str] = []
    chunk_size = 256
    for start in range(0, len(expected_ids), chunk_size):
        ids = expected_ids[start:start + chunk_size]
        batch_articles = articles[start:start + chunk_size]
        expected_hashes = {
            point_id: embedding_text_sha256(article)
            for article, point_id in zip(batch_articles, ids)
        }
        present = _retrieve_points(client, collection, ids, start, start + len(ids))
        valid_ids = {
            str(point.id)
            for point in present
            if (point.payload or {}).get("embedding_text_sha256") == expected_hashes.get(str(point.id))
            and (point.payload or {}).get("embedding_backend") == config.backend
            and (point.payload or {}).get("embedding_model") == _embedding_model_name(config)
        }
        missing.extend([point_id for point_id in ids if point_id not in valid_ids])

    if missing:
        _MISSING_IDS_LOG.write_text(
            json.dumps({"collection": collection, "missing_ids": missing}, indent=2),
            encoding="utf-8",
        )
        print(f"  FAIL missing/stale ids: {len(missing)} (see {_MISSING_IDS_LOG})")
        return False

    if count != expected:
        print("  FAIL count mismatch even though all expected ids are present")
        return False

    print("  OK all expected chunk ids are present")
    return True


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Setup/test configured embedder + Qdrant")
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--force", action="store_true", help="Recreate the collection")
    parser.add_argument("--test-only", action="store_true", help="Only connectivity tests; skip corpus indexing")
    parser.add_argument("--workers", type=int, default=1, help="Parallel embed/upsert workers")
    parser.add_argument(
        "--max-resume-passes",
        type=int,
        default=3,
        help="Automatically retry incomplete/failed indexing passes in the same command",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print(f"  {config.backend} embedding + Qdrant setup")
    print("=" * 60)

    print("\n[1/7] Validating config")
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"  FAIL {error}")
        sys.exit(1)
    print("  OK config")

    embedder, rewriter = _make_embedding_components(config)

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
    rewritten = rewriter.rewrite_strict("Nguoi lao dong duoc nghi bao nhieu ngay phep nam?")
    if not rewritten.get("rewritten_query") or not rewritten.get("topic_description"):
        raise RuntimeError("query rewrite preflight returned incomplete output")
    print(f"  OK rewritten_query: {rewritten['rewritten_query'][:80]}")

    if args.test_only:
        print("\n[5/7] --test-only set; skipping corpus indexing")
        print("\nDone (test-only).")
        return

    _acquire_collection_lock(config.qdrant.collection)

    print(f"\n[5/7] Indexing corpus from {args.corpus}")
    _ensure_collection(client, config, force=args.force)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    articles = corpus["articles"]
    if corpus.get("total_articles") != len(articles):
        raise ValueError("corpus total_articles does not match articles length")
    chunk_ids = [article.get("chunk_id") for article in articles]
    if any(not chunk_id for chunk_id in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("corpus contains missing or duplicate chunk_id values")
    for chunk_id in chunk_ids:
        uuid.UUID(str(chunk_id))
    if any(not article.get("relevant_article_str") for article in articles):
        raise ValueError("corpus contains empty relevant_article_str values")
    print(f"  {len(articles)} articles loaded")
    _PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    _PROGRESS_LOG.unlink(missing_ok=True)
    _ERROR_LOG.unlink(missing_ok=True)
    _MISSING_IDS_LOG.unlink(missing_ok=True)
    verified = False
    last_error: Exception | None = None
    for pass_number in range(1, max(1, args.max_resume_passes) + 1):
        print(f"\n  indexing pass {pass_number}/{max(1, args.max_resume_passes)}")
        try:
            indexed = _index_corpus_parallel(config, articles, workers=args.workers)
            print(f"  OK checked/indexed {indexed} articles into '{config.qdrant.collection}'")
            print("\n[6/7] Verifying full collection")
            verified = _verify_complete_collection(client, config, articles)
            if verified:
                break
        except Exception as exc:
            last_error = exc
            print(f"  pass {pass_number} failed: {type(exc).__name__}: {exc}")
        if pass_number < max(1, args.max_resume_passes):
            time.sleep(min(60, 5 * (2 ** (pass_number - 1))))
    if not verified:
        if last_error is not None:
            raise RuntimeError(
                f"collection incomplete after {args.max_resume_passes} passes"
            ) from last_error
        raise SystemExit(2)

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
