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
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.config import RetrievalConfig, load_config
from vertex_backends import VertexEmbedder, VertexQueryRewriter


def _validate_config(config: RetrievalConfig) -> list[str]:
    errors: list[str] = []
    if not config.qdrant.url:
        errors.append("qdrant.url is empty — set QDRANT_URL in .env")
    if not config.qdrant.api_key:
        errors.append("qdrant.api_key is empty — set QDRANT_API_KEY in .env")
    if not config.vllm.gcp_project and not os.environ.get("GOOGLE_API_KEY"):
        errors.append(
            "No GCP auth: set GCP_PROJECT (Vertex/ADC) OR GOOGLE_API_KEY in .env"
        )
    return errors


def _qdrant_client(config: RetrievalConfig):
    from qdrant_client import QdrantClient
    return QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)


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
    errors = _validate_config(config)
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
    n = embedder.embed_corpus(articles, qdrant_client=client)
    print(f"  ✓ indexed {n} articles into '{config.qdrant.collection}'")

    # 6. Test search
    print("\n[6/6] Test search: 'doanh nghiệp nhỏ và vừa' (top 5)…")
    q_vec = embedder.embed_query("doanh nghiệp nhỏ và vừa").tolist()
    hits = client.search(
        collection_name=config.qdrant.collection, query_vector=q_vec, limit=5
    )
    for rank, h in enumerate(hits, 1):
        p = h.payload or {}
        print(f"  {rank}. [{p.get('law_id','')}] {p.get('dieu_number','')} "
              f"— {str(p.get('dieu_title',''))[:50]}  (score={h.score:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
