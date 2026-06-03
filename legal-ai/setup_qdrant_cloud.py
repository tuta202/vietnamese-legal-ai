"""
setup_qdrant_cloud.py — validate + bootstrap the Vertex AI / Qdrant Cloud stack.

Uses the unified config (retrieval.config.load_config) and the Vertex subclasses
from vertex_backends — the same components the pipeline uses.

Workflow:
  1. Validate config (qdrant.url/api_key filled; GCP auth available).
  2. Test Gemini embedding (embed 1 sentence, assert dimension matches config).
  3. Test Qdrant Cloud connection.
  4. Test Gemini LLM (rewrite 1 question).
  5. Load corpus → embed → upsert into Qdrant Cloud.
  6. Test search (1 query, show top 5).

CLI:
    python setup_qdrant_cloud.py --config config_vertex.yaml \
        --corpus corpus/data/corpus.json [--force] [--test-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.config import RetrievalConfig, load_config, validate_config
from vertex_backends import VertexEmbedder, VertexQueryRewriter


def _qdrant_client(config: RetrievalConfig):
    from qdrant_client import QdrantClient
    # Generous timeout — the free Qdrant Cloud tier can be slow on upserts.
    return QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key, timeout=120)


def _index_corpus(client, embedder, config: RetrievalConfig, articles: list[dict]) -> int:
    """
    Resilient batched embed + upsert (wait=False, per-batch retry).
    Idempotent (point id = chunk_id), so re-running fills any gaps without
    re-embedding completed batches within a run.
    """
    import time
    import uuid
    from qdrant_client.models import PointStruct

    bs = max(1, int(config.embedding.batch_size))
    total = len(articles)
    done = 0
    for i in range(0, total, bs):
        batch = articles[i:i + bs]
        texts = [embedder._format_document(a) for a in batch]
        vectors = embedder.embed_documents(texts)
        points = [
            PointStruct(
                id=str(uuid.UUID(a["chunk_id"])),
                vector=vec.tolist(),
                payload={
                    "chunk_id":             a["chunk_id"],
                    "law_id":               a["law_id"],
                    "law_type":             a["law_type"],
                    "law_name":             a["law_name"],
                    "dieu_number":          a["dieu_number"],
                    "dieu_title":           a["dieu_title"],
                    "content":              a["content"],
                    "relevant_doc_str":     a["relevant_doc_str"],
                    "relevant_article_str": a["relevant_article_str"],
                },
            )
            for a, vec in zip(batch, vectors)
        ]
        for attempt in range(4):
            try:
                client.upsert(collection_name=config.qdrant.collection,
                              points=points, wait=False)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait_s = 2 * (attempt + 1)
                print(f"\n  [retry] upsert batch failed ({type(e).__name__}); "
                      f"retrying in {wait_s}s…")
                time.sleep(wait_s)
        done += len(points)
        print(f"  upserted {done}/{total}", end="\r", flush=True)
    print()
    return done


def _ensure_collection(client, config: RetrievalConfig, force: bool) -> None:
    from qdrant_client.models import Distance, VectorParams

    name = config.qdrant.collection
    distance = {
        "cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID,
    }.get(str(config.qdrant.distance).lower(), Distance.COSINE)

    existing = {c.name for c in client.get_collections().collections}
    if force and name in existing:
        client.delete_collection(name)
        existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=config.qdrant.vector_size, distance=distance),
        )
        print(f"  Created collection '{name}' ({config.qdrant.vector_size}d, {distance})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Setup/test Vertex + Qdrant Cloud")
    parser.add_argument("--config", default="config_vertex.yaml")
    parser.add_argument("--corpus", default="corpus/data/corpus.json")
    parser.add_argument("--force",  action="store_true", help="Recreate the collection")
    parser.add_argument("--test-only", action="store_true",
                        help="Only connectivity tests; skip corpus indexing")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("  Vertex AI + Qdrant Cloud setup")
    print("=" * 60)

    # 1. Validate config
    print("\n[1/6] Validating config…")
    errors = validate_config(config)
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print("  ✓ config OK")

    embedder = VertexEmbedder(config, dry_run=False)
    rewriter = VertexQueryRewriter(config, mock=False)

    # 2. Gemini embedding
    print("\n[2/6] Testing Gemini embedding…")
    vec = embedder.embed_query("Doanh nghiệp nhỏ và vừa là gì?")
    assert len(vec) == config.embedding.dimension, (
        f"dimension mismatch: got {len(vec)}, expected {config.embedding.dimension}"
    )
    print(f"  ✓ embedding dim = {len(vec)}")

    # 3. Qdrant Cloud connection
    print("\n[3/6] Testing Qdrant Cloud connection…")
    client = _qdrant_client(config)
    collections = client.get_collections().collections
    print(f"  ✓ connected — {len(collections)} existing collection(s)")

    # 4. Gemini LLM
    print("\n[4/6] Testing Gemini LLM (query rewrite)…")
    rw = rewriter.rewrite("Người lao động được nghỉ bao nhiêu ngày phép năm?")
    print(f"  ✓ rewritten_query: {rw['rewritten_query'][:80]}")

    if args.test_only:
        print("\n[5/6] --test-only set → skipping corpus indexing")
        print("\nDone (test-only).")
        return

    # 5. Index corpus
    print(f"\n[5/6] Indexing corpus from {args.corpus}…")
    _ensure_collection(client, config, force=args.force)
    articles = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["articles"]
    print(f"  {len(articles)} articles loaded")
    n = _index_corpus(client, embedder, config, articles)
    print(f"  ✓ indexed {n} articles into '{config.qdrant.collection}'")

    # 6. Test search
    print("\n[6/6] Test search: 'doanh nghiệp nhỏ và vừa' (top 5)…")
    q_vec = embedder.embed_query("doanh nghiệp nhỏ và vừa").tolist()
    resp = client.query_points(
        collection_name=config.qdrant.collection, query=q_vec, limit=5,
        with_payload=True,
    )
    for rank, h in enumerate(resp.points, 1):
        p = h.payload or {}
        print(f"  {rank}. [{p.get('law_id','')}] {p.get('dieu_number','')} "
              f"— {str(p.get('dieu_title',''))[:50]}  (score={h.score:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
