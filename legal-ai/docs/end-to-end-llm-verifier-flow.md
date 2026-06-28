# RAG Workflow

Workflow hiện tại dùng `corpus_clean`, global RRF, intent retrieval, BGE
intent-wise rerank, rồi LLM verifier.

```mermaid
flowchart TD
    Q["Original question"]

    Q --> ANALYZER["Query analyzer"]
    ANALYZER --> RQ["rewritten_query"]
    ANALYZER --> TD["topic_description"]
    ANALYZER --> INTENTS["legal_intents[]"]

    subgraph GLOBAL["Global retrieval"]
        RQ --> BM25Q["BM25(rewritten_query)"]
        RQ --> DENSEQ["Dense(rewritten_query)"]
        TD --> DENSET["Dense(topic_description)"]
        BM25Q --> RRF["RRF fusion"]
        DENSEQ --> RRF
        DENSET --> RRF
        RRF --> TOP60["top60_rrf"]
    end

    subgraph INTENT["Intent retrieval"]
        INTENTS --> EACHINTENT["For each legal_intent"]
        EACHINTENT --> IBM25["BM25(intent)"]
        EACHINTENT --> IDENSE["Dense(intent)"]
        IBM25 --> IRRF["RRF(intent)"]
        IDENSE --> IRRF
        IRRF --> ITOP10["top10_each_intent"]
        ITOP10 --> IHITS["intent_hits"]
    end

    TOP60 --> KEEP["keep = top60_rrf | intent_hits"]
    IHITS --> KEEP

    subgraph BGE["BGE intent-wise rerank"]
        KEEP --> DOCS["Hydrate articles from corpus_clean"]
        INTENTS --> BGERERANK["Rerank keep for each legal_intent"]
        DOCS --> BGERERANK
        BGERERANK --> BGECACHE["BGE ranked articles per intent"]
    end

    subgraph UNION["Recall-preserving candidate union"]
        TOP60 --> RRF8["top8_rrf_global"]
        BGECACHE --> BGE3["top3_bge_each_intent"]
        IHITS --> IH3["top3_intent_hits_each_intent"]
        RRF8 --> CANDS["candidate_articles"]
        BGE3 --> CANDS
        IH3 --> CANDS
    end

    subgraph VERIFY["LLM verifier"]
        CANDS --> LLMINPUT["question + candidate_articles only"]
        LLMINPUT --> LLMV["LLM verifies necessary articles"]
        LLMV --> RESCUE["adaptive fallback / rescue"]
        RESCUE --> FINAL["final_article_ids"]
    end

    FINAL --> SUBMISSION["submission.zip"]
```

Core rules:

```text
Corpus source: corpus/data/corpus_clean.json
Qdrant collection: legal_vn_vertex_clean_v1
BM25 index: retrieval/data/bm25_index_clean_v1.pkl

keep = top60_rrf | intent_hits

candidate_articles =
    top8_rrf_global
  | top3_bge_each_intent
  | top3_intent_hits_each_intent

LLM verifier receives only question + candidate article content.
It does not receive retrieval scores, ranks, sources, rewritten_query, or intents.
```
