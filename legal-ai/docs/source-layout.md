# Source Layout

The repository separates executable source code from runtime data and generated artifacts.

```text
legal-ai/
├── legal_rag/
│   ├── common/          shared paths, Qdrant factory, retries, article lookup
│   ├── backends/        Vertex/Gemini, GPU/Qwen-Gemma, shared BGE adapters
│   ├── generation/      answer prompt and generator
│   ├── retrieval/       query analysis, global RRF, per-intent RRF
│   ├── ranking/         intent-wise BGE and tiered candidate union
│   ├── verification/    Stage 1, cleanup, Stage 2, final verifier, role gate
│   ├── indexing/        BM25 and Qdrant collection builders
│   ├── orchestration/   strict resumable end-to-end workflow
│   ├── output/          submission validation and packaging
│   └── pipeline.py      reusable online retrieval/generation pipeline
├── corpus/              corpus preparation tools and gitignored corpus data
├── retrieval/data/      gitignored BM25 indexes
├── cache/               resumable stage caches
├── outputs/             run artifacts, diagnostics, logs, submissions
├── tests/               all automated tests
├── docs/                workflow and maintenance documentation
├── config_gpu_clean.yaml
└── config_vertex_clean.yaml
```

## Public Commands

The root scripts are compatibility wrappers. Existing commands remain stable:

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py --config config_vertex_clean.yaml ...
.\.venv\Scripts\python.exe setup_qdrant_cloud.py --config config_gpu_clean.yaml ...
.\.venv\Scripts\python.exe build_core_bm25.py ...
.\.venv\Scripts\python.exe pipeline.py --config config_vertex_clean.yaml ...
```

Internal stages are package modules and are normally invoked by the orchestrator:

```text
legal_rag.retrieval.query_analysis
legal_rag.retrieval.global_rrf
legal_rag.retrieval.intent_rrf
legal_rag.ranking.intent_bge
legal_rag.ranking.candidate_union
legal_rag.verification.candidate_verifier
legal_rag.verification.deterministic_cleanup
legal_rag.verification.final_collective
legal_rag.verification.enforcement_gate
```

## Dependency Direction

```text
common <- backends/retrieval/generation
retrieval + backends <- ranking
common + retrieval + backends <- verification
all stages <- orchestration
```

Runtime directories (`corpus/data`, `retrieval/data`, `cache`, `outputs`) must never be
imported as Python packages. Model failures remain uncached and are retried by the
orchestrator; moving modules does not change cache or submission schemas.
