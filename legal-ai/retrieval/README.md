# Retrieval Stack — legal-ai/retrieval/

Hybrid BM25 + dense retrieval with RRF fusion for Vietnamese legal Q&A.

## Modules

| Module | Role | Offline-safe |
|---|---|---|
| `config.py` | Load `config.yaml` into `RetrievalConfig` | ✅ |
| `bm25_index.py` | Okapi BM25 over 1044 articles | ✅ |
| `embedder.py` | Dense embedder (Qwen3-Embedding-8B) | ✅ `dry_run=True` |
| `query_rewriter.py` | LLM query rewrite + HyDE (vLLM) | ✅ `mock=True` |
| `reranker.py` | Cross-encoder reranker (Qwen3-Reranker-4B) | ✅ `mock=True` |
| `hybrid_search.py` | Orchestrates all 5 pipeline steps | ✅ (with mocks) |

## Offline testing (no infra required)

```bash
# From legal-ai/ directory
pip install pytest pyyaml numpy

# Run all 27 tests (~0.4s)
pytest retrieval/tests/ -v

# Build BM25 index on real corpus (no GPU)
python retrieval/bm25_index.py
# → retrieval/data/bm25_index.pkl
```

## Setup infra (Qdrant + vLLM)

```bash
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Create collection (run once)
python - <<'EOF'
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
c = QdrantClient("localhost", port=6333)
c.create_collection("legal_vn", vectors_config=VectorParams(size=4096, distance=Distance.COSINE))
EOF

# vLLM (requires GPU)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

## Embed corpus into Qdrant

```python
import json
from retrieval.config import load_config
from retrieval.embedder import LegalEmbedder
from qdrant_client import QdrantClient

cfg      = load_config("retrieval/config.yaml")
articles = json.loads(open("corpus/data/corpus.json").read())["articles"]
client   = QdrantClient(cfg.qdrant.host, port=cfg.qdrant.port)

embedder = LegalEmbedder(cfg)           # loads model on first call
embedder.embed_corpus(articles, client) # upserts 1044 points
```

## Integration test (with infra up)

```python
from retrieval.bm25_index import BM25Index
from retrieval.config import load_config
from retrieval.embedder import LegalEmbedder
from retrieval.hybrid_search import HybridSearcher
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import LegalReranker
from qdrant_client import QdrantClient

cfg      = load_config("retrieval/config.yaml")
bm25     = BM25Index.load("retrieval/data/bm25_index.pkl")
embedder = LegalEmbedder(cfg)
rewriter = QueryRewriter(cfg)
reranker = LegalReranker(cfg)
qdrant   = QdrantClient(cfg.qdrant.host, port=cfg.qdrant.port)

from retrieval.hybrid_search import HybridSearcher
searcher = HybridSearcher(cfg, bm25, embedder, rewriter, qdrant)
results  = searcher.search("Công ty tôi có 5 người, có phải DNNVV không?")
reranked = reranker.rerank("...", results)
for r in reranked:
    print(r["relevant_article_str"])
```

## Config reference

```yaml
# retrieval/config.yaml
qdrant:
  host: localhost
  port: 6333
  collection: legal_vn
vllm:
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-7B-Instruct
models:
  embedder: Qwen/Qwen3-Embedding-8B
  reranker: Qwen/Qwen3-Reranker-4B
retrieval:
  top_k_dense: 20
  top_k_bm25: 20
  top_k_fusion: 20
  top_k_rerank: 7
  rrf_k: 60
```
