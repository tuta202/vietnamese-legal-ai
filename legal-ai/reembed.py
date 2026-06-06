"""
reembed.py — one-shot corpus re-embedding when the embedder changes.

Recreates the Qdrant collection at the config's dimension and embeds every
article in corpus.json with the configured embedder (Garden Qwen3-Embedding-8B by
default), upserting the SAME payload shape the pipeline expects. Resumable via a
checkpoint file so an interrupted ~2h run continues where it stopped.

This overlaps with setup_qdrant_cloud.py (which targets Qdrant Cloud / Gemini);
reembed.py is the local-Qdrant + Garden-embedder counterpart with an explicit
checkpoint and a self-retrieval verification step.

CLI:
    python reembed.py --config config_garden.yaml --fresh    # wipe + embed all
    python reembed.py --config config_garden.yaml --resume   # continue from checkpoint
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.config import load_config, validate_config

_CHECKPOINT = _ROOT / "reembed_checkpoint.json"
_DEFAULT_CORPUS = _ROOT / "corpus" / "data" / "corpus.json"
_CHECKPOINT_EVERY = 1000   # articles between checkpoint writes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qdrant_client(config):
    from qdrant_client import QdrantClient
    q = config.qdrant
    if q.url:
        return QdrantClient(url=q.url, api_key=q.api_key, timeout=120)
    return QdrantClient(host=q.host, port=q.port, timeout=120)


def _ensure_collection(client, config, *, recreate: bool) -> None:
    from qdrant_client.models import Distance, VectorParams
    name = config.qdrant.collection
    distance = {
        "cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID,
    }.get(str(config.qdrant.distance).lower(), Distance.COSINE)

    existing = {c.name for c in client.get_collections().collections}
    if recreate and name in existing:
        client.delete_collection(name)
        existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config.qdrant.vector_size, distance=distance
            ),
        )
        print(f"  created collection '{name}' "
              f"({config.qdrant.vector_size}d, {distance})")


def _payload(a: dict) -> dict:
    return {
        "chunk_id":             a["chunk_id"],
        "law_id":               a["law_id"],
        "law_type":             a["law_type"],
        "law_name":             a["law_name"],
        "dieu_number":          a["dieu_number"],
        "dieu_title":           a["dieu_title"],
        "content":              a["content"],
        "relevant_doc_str":     a["relevant_doc_str"],
        "relevant_article_str": a["relevant_article_str"],
    }


def _load_checkpoint(collection: str) -> int:
    if not _CHECKPOINT.exists():
        return 0
    try:
        data = json.loads(_CHECKPOINT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if data.get("collection") != collection:
        print(f"  checkpoint is for a different collection "
              f"({data.get('collection')!r}) — ignoring")
        return 0
    return int(data.get("last_offset", 0))


def _save_checkpoint(collection: str, offset: int) -> None:
    _CHECKPOINT.write_text(
        json.dumps({
            "collection": collection,
            "last_offset": offset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def _verify(client, embedder, config, articles: list[dict]) -> bool:
    name = config.qdrant.collection
    expected = len(articles)
    count = client.count(collection_name=name, exact=True).count
    print(f"  point count: {count} (expected {expected})")
    if count != expected:
        print("  ✗ count mismatch")
        return False

    # Self-retrieval: a handful of articles, embedded as documents, should rank
    # themselves #1 when searched.
    sample = random.sample(articles, min(5, len(articles)))
    rank1 = 0
    for a in sample:
        vec = embedder.embed_documents([embedder._format_document(a)])[0]
        resp = client.query_points(
            collection_name=name, query=vec.tolist(), limit=1, with_payload=True
        )
        top = resp.points[0] if resp.points else None
        hit = top and (top.payload or {}).get("chunk_id") == a["chunk_id"]
        rank1 += 1 if hit else 0
        print(f"    [{a['chunk_id'][:8]}] self-retrieval rank-1: {'✓' if hit else '✗'}")
    ok = rank1 >= max(1, int(0.8 * len(sample)))
    print(f"  self-retrieval rank-1: {rank1}/{len(sample)} {'✓' if ok else '✗'}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Re-embed corpus into Qdrant")
    ap.add_argument("--config", default="config_garden.yaml")
    ap.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true",
                      help="wipe the collection + checkpoint, embed from scratch")
    mode.add_argument("--resume", action="store_true",
                      help="continue from the last checkpoint")
    args = ap.parse_args()

    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        print("Invalid config:")
        for p in problems:
            print(f"  ✗ {p}")
        sys.exit(1)

    from garden_backends import make_garden_components
    embedder = make_garden_components(config, mock=False)[0]

    from qdrant_client.models import PointStruct

    name = config.qdrant.collection
    articles = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["articles"]
    total = len(articles)
    bs = max(1, int(config.embedding.batch_size))
    print(f"Corpus: {total} articles  |  collection: {name}  |  "
          f"dim: {config.embedding.dimension}  |  batch: {bs}")

    client = _qdrant_client(config)
    _ensure_collection(client, config, recreate=args.fresh)

    start_offset = 0
    if args.resume:
        start_offset = _load_checkpoint(name)
        print(f"  resuming from offset {start_offset}")
    elif args.fresh:
        _CHECKPOINT.unlink(missing_ok=True)

    try:
        from tqdm import tqdm
        bar = tqdm(total=total, initial=start_offset, unit="art")
    except ImportError:
        bar = None

    t0 = time.time()
    done = start_offset
    last_ckpt = start_offset
    for i in range(start_offset, total, bs):
        batch = articles[i:i + bs]
        texts = [embedder._format_document(a) for a in batch]
        vectors = embedder.embed_documents(texts)
        points = [
            PointStruct(id=str(uuid.UUID(a["chunk_id"])),
                        vector=vec.tolist(), payload=_payload(a))
            for a, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=name, points=points, wait=False)

        done += len(batch)
        if bar:
            bar.update(len(batch))
        else:
            rate = (done - start_offset) / max(1e-9, time.time() - t0)
            eta = (total - done) / rate if rate else 0
            print(f"  {done}/{total}  ({rate:.1f} art/s, ETA {eta/60:.0f} min)",
                  end="\r", flush=True)

        if done - last_ckpt >= _CHECKPOINT_EVERY:
            _save_checkpoint(name, done)
            last_ckpt = done

    _save_checkpoint(name, done)
    if bar:
        bar.close()
    else:
        print()
    print(f"Embedded {done - start_offset} new articles in "
          f"{(time.time() - t0) / 60:.1f} min")

    print("Verifying…")
    ok = _verify(client, embedder, config, articles)
    if not ok:
        print("✗ verification FAILED")
        sys.exit(2)
    print("✓ re-embed complete and verified")


if __name__ == "__main__":
    main()
