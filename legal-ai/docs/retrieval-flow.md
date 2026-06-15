# Luồng Retrieval hiện tại

Tài liệu này mô tả luồng retrieval đang chạy trong hệ thống Vietnamese Legal AI sau khi bổ sung nhánh **Legal Intent Decomposition**.

Mục tiêu thiết kế hiện tại:

- Giữ nguyên luồng tốt cũ: `rewrite -> global retrieval -> RRF -> BGE scoring/rerank/package`.
- Không sửa prompt hoặc logic của `QueryRewriter`.
- Thêm một call LLM riêng để tách intent pháp lý, dùng làm nhánh cứu recall.
- Không đưa intent results vào main global RRF ở giai đoạn này.

Các file chính:

- `pipeline.py`
- `retrieval/query_rewriter.py`
- `retrieval/intent_decomposer.py`
- `retrieval/bm25_index.py`
- `retrieval/embedder.py`
- `retrieval/hybrid_search.py`
- `bge_ab_probe.py`

---

## 1. Tổng quan

Luồng hiện tại có hai nhánh retrieval:

```text
Original question
  |
  +--> Existing Rewrite Chain
  |      |
  |      +--> rewritten_query
  |      +--> topic_description
  |      |
  |      +--> Global Retrieval
  |             - BM25(rewritten_query)
  |             - Dense(rewritten_query)
  |             - Dense(topic_description)
  |             |
  |             +--> RRF
  |             +--> rrf_top200
  |
  +--> Legal Intent Decomposition Chain
         |
         +--> intents: list[str]
         |
         +--> Intent Retrieval
                for each intent:
                  - BM25(intent, top50)
                  - Dense(intent, top50)
                  - RRF(intent)
                  - keep top10
                |
                +--> intent_hits
```

Điểm quan trọng:

- `rrf_top200` là output chính của global retrieval, giống luồng cũ.
- `intent_hits` là nhánh phụ, chỉ dùng để mở rộng keep set khi đánh giá recall.
- Intent hits không thay thế `rewritten_query`.
- Intent hits không đi vào main RRF global.
- Intent hits không làm thay đổi thứ tự RRF của `rrf_top200`.

---

## 2. Existing Rewrite Chain

File: `retrieval/query_rewriter.py`

Input:

```text
original_question
```

Output:

```python
{
    "rewritten_query": str,
    "topic_description": str,
}
```

Luồng này giữ nguyên như trước. Prompt rewrite hiện tại vẫn chỉ có nhiệm vụ:

- tạo `rewritten_query` cô đọng, tối ưu cho global retrieval;
- tạo `topic_description` mô tả chủ đề, thuật ngữ, loại văn bản liên quan;
- không sinh intent;
- không quyết định multi-hop;
- không bị gộp chung với decomposition prompt.

Fallback:

- Nếu LLM lỗi hoặc JSON parse lỗi, `rewritten_query` fallback về câu hỏi gốc.
- `topic_description` có thể rỗng.
- Pipeline vẫn chạy tiếp.

---

## 3. Legal Intent Decomposition Chain

File: `retrieval/intent_decomposer.py`

Đây là một chain riêng, gọi LLM thêm một lần sau rewrite.

Input:

```text
original_question
```

Output:

```python
@dataclass
class IntentAnalysis:
    intents: list[str]
```

Không dùng:

- `LegalIntent`
- `intent_id`
- `priority`
- `legal_area`
- `is_multihop`

Lý do: output càng đơn giản thì LLM càng ổn định, dễ parse, ít làm hỏng pipeline.

### 3.1 Định nghĩa intent

Mỗi intent là một truy vấn pháp lý độc lập, có thể dùng trực tiếp cho:

```text
BM25(intent)
Dense(intent)
```

Intent tốt:

```json
[
  "xác định hành vi xâm phạm quyền tác giả đối với phần mềm",
  "xác định thiệt hại và tổn thất cơ hội kinh doanh do hành vi xâm phạm quyền tác giả",
  "hồ sơ tài liệu chứng cứ khi yêu cầu xử lý hành vi xâm phạm quyền tác giả"
]
```

Intent không tốt:

```json
[
  "quyền tác giả",
  "thiệt hại",
  "hồ sơ"
]
```

Vì các chuỗi trên chỉ là keyword/topic label, không phải retrieval query độc lập.

### 3.2 Prompt decomposition

Prompt decomposition yêu cầu LLM trả JSON duy nhất:

```json
{
  "intents": [
    "..."
  ]
}
```

Rules chính:

- Luôn trả ít nhất 1 intent.
- Câu đơn giản/single-hop: thường trả 1 intent.
- Câu multi-hop: trả 2-6 intents.
- Mỗi intent phải là một retrieval query đầy đủ.
- Không output keyword-only.
- Không copy nguyên văn toàn bộ câu hỏi.
- Không thêm giải thích.
- Không đưa tên luật nếu câu hỏi không nêu rõ hoặc không strongly implied.

Fallback:

- Nếu LLM lỗi hoặc parse lỗi, `intents = [original_question]`.
- Điều này giữ pipeline không bị gãy.

---

## 4. Global Retrieval

File chính: `pipeline.py::step_retrieve`

Global retrieval dùng output của rewrite chain:

```python
query = state.rewritten_query or state.question
```

Sau đó chạy:

```python
bm25_hits = BM25(rewritten_query, top_k_bm25)
dense_hits = Dense(rewritten_query, top_k_dense)
dense_topic_hits = Dense(topic_description, top_k_dense)  # nếu topic khác query
```

Các ranked list được fusion bằng RRF:

```python
rrf_top200 = rrf_fusion(
    [
        bm25_hits,
        dense_hits,
        dense_topic_hits,
    ],
    k=rrf_k,
)[:top_k_fusion]
```

Với config hiện tại:

```yaml
retrieval:
  top_k_dense: 240
  top_k_bm25: 240
  top_k_fusion: 200
  rrf_k: 60
```

Kết quả:

```python
state.fused_results = global_results
```

Invariant quan trọng:

```text
state.fused_results chỉ chứa global RRF results.
Intent hits không được append vào state.fused_results.
```

Điều này giữ nguyên behavior của luồng cũ.

---

## 5. Intent Retrieval

Intent retrieval chạy sau khi đã có `state.intent_queries`.

Với mỗi intent:

```python
bm25_i = BM25(intent, top_k=50)
dense_i = Dense(intent, top_k=50)

intent_rrf = rrf_fusion(
    [
        bm25_i,
        dense_i,
    ],
    k=rrf_k,
)

intent_hits |= intent_rrf[:10]
```

Trong code:

- `pipeline.py::_retrieve_intent_hits`
- output lưu ở `state.intent_hits`

Config:

```yaml
retrieval:
  enable_intent_retrieval: true
  intent_top_k_bm25: 50
  intent_top_k_dense: 50
  intent_top_k_rrf: 10
```

Metadata gắn trên intent hits:

```python
{
    "retrieval_source": "intent",
    "from_intent": True,
    "intent_ids": [1, ...],
    "intent_queries": ["...", ...],
    "intent_rank": int,
    "intent_rrf_score": float,
}
```

Các metadata này chỉ để audit/debug. Data model chính của decomposer vẫn chỉ là `list[str]`.

---

## 6. Candidate Keep Set

Luồng đánh giá hiện tại dựa trên BGE probe và leaderboard.

Old keep:

```python
keep_old = top80_bge | top80_rrf
```

New keep:

```python
keep_new = top80_bge | top80_rrf | intent_hits
```

Trong đó:

- `top80_bge`: lấy từ BGE score trên global RRF pool.
- `top80_rrf`: lấy theo thứ tự global RRF pool.
- `intent_hits`: top10 mỗi intent từ nhánh intent retrieval.

Điểm quan trọng:

- `keep_new` là set.
- Không sort toàn bộ keep.
- Không lấy topK từ keep.
- Mục tiêu của keep là tăng recall evidence coverage trước bước final verification/ranking.

---

## 7. BGE Probe Integration

File: `bge_ab_probe.py`

Per question, probe chạy:

```text
step_rewrite
step_decompose
step_retrieve
BGE score trên state.fused_results
```

Điểm cần nhớ:

- BGE vẫn score trên `state.fused_results`, tức global RRF pool cũ.
- `intent_hits` không được BGE score ở bước này.
- Output JSON lưu thêm `intent_hits` riêng.

Shape output chính:

```json
{
  "id": 1,
  "question": "...",
  "rewritten_query": "...",
  "topic_description": "...",
  "intents": ["..."],
  "num_intents": 3,
  "retrieval_metrics": {
    "num_intents": 3,
    "intent_lengths": [64, 82, 71],
    "global_count": 200,
    "intent_count": 25,
    "keep_count": 214,
    "global_recall": null,
    "intent_only_recall": null,
    "keep_recall": null,
    "cross_document": null
  },
  "intent_hits": [
    {
      "article": "...",
      "doc": "...",
      "intent_ids": [1],
      "intent_queries": ["..."]
    }
  ],
  "pool": [
    {
      "article": "...",
      "doc": "...",
      "rrf": 0.03125,
      "bge": 0.78231
    }
  ]
}
```

Khi package/analyze:

```python
rrf_order = pool
bge_order = sorted(pool, key=lambda c: c["bge"], reverse=True)
intent_order = entry["intent_hits"]

submission_union = top80_bge + top80_rrf + intent_order
```

Dedup được xử lý khi tạo `results.json`.

---

## 8. Logging Và Metrics

Pipeline lưu các thông tin sau khi có `scores-detail` hoặc khi chạy BGE probe:

```python
{
    "question_id": ...,
    "original_question": ...,
    "rewritten_query": ...,
    "topic_description": ...,
    "intents": [...],
    "num_intents": len(intents),
    "global_candidate_count": len(state.fused_results),
    "intent_hit_count": len(state.intent_hits),
    "keep_count": len(global ∪ intent)
}
```

Nếu có labels/gold, có thể tính thêm:

```python
{
    "global_recall": ...,
    "intent_only_recall": ...,
    "keep_recall": ...,
    "gold_recovered_by_intents": [...]
}
```

Trong leaderboard input hiện tại không có gold labels, nên các trường recall để `null`.

---

## 9. BM25

File: `retrieval/bm25_index.py`

BM25 là lexical retrieval in-process:

- index trên `dieu_title + content`;
- tokenizer đơn giản cho tiếng Việt;
- dùng Okapi BM25;
- output là list `(chunk_id, score)`.

BM25 được dùng ở hai nơi:

```text
Global: BM25(rewritten_query, top_k_bm25=240)
Intent: BM25(intent, top_k=50)
```

---

## 10. Dense Retrieval

File:

- `retrieval/embedder.py`
- `gpu_backends.py::GpuEmbedder`
- `vertex_backends.py::VertexEmbedder`
- `pipeline.py::_dense_search`

Dense retrieval dùng Qwen3/Gemini embedding tùy backend.

Global dense:

```text
Dense(rewritten_query, top_k_dense=240)
Dense(topic_description, top_k_dense=240)
```

Intent dense:

```text
Dense(intent, top_k=50)
```

`_dense_search` có tham số `limit` tùy nhánh:

- không truyền `limit` -> dùng `top_k_dense`;
- intent retrieval truyền `limit=intent_top_k_dense`.

---

## 11. RRF

File: `retrieval/hybrid_search.py::rrf_fusion`

Công thức:

```python
score[chunk_id] += 1 / (rrf_k + rank)
```

Vì dùng rank thay vì raw score, RRF ghép ổn định giữa:

- BM25 score;
- dense cosine score;
- các list từ nhiều query khác nhau.

Global RRF:

```text
BM25(rewritten) + Dense(rewritten) + Dense(topic)
```

Intent RRF:

```text
BM25(intent) + Dense(intent)
```

Hai loại RRF này độc lập.

---

## 12. Fallbacks

Rewrite fallback:

```text
rewritten_query = original_question
topic_description = ""
```

Decompose fallback:

```text
intents = [original_question]
```

Retrieval fallback:

Nếu global retrieval rỗng, chạy lại:

```python
BM25(original_question, top_k_fusion)
```

Các fallback này bảo đảm pipeline không bị abort vì một lỗi LLM hoặc retrieval tạm thời.

---

## 13. Bản đồ code

| Chức năng | File |
|---|---|
| Orchestrator chính | `pipeline.py` |
| Rewrite chain cũ | `retrieval/query_rewriter.py` |
| Intent decomposition chain mới | `retrieval/intent_decomposer.py` |
| Global retrieval | `pipeline.py::step_retrieve` |
| Intent retrieval | `pipeline.py::_retrieve_intent_hits` |
| Dense search helper | `pipeline.py::_dense_search` |
| RRF | `retrieval/hybrid_search.py::rrf_fusion` |
| BM25 | `retrieval/bm25_index.py` |
| BGE pool + union packaging | `bge_ab_probe.py` |

---

## 14. Invariants Cần Giữ

Các invariant này là phần quan trọng nhất của thiết kế hiện tại:

1. `QueryRewriter` không bị sửa prompt để sinh intents.
2. `LegalIntentDecomposer` là call LLM riêng, input là original question.
3. `state.fused_results` chỉ chứa global RRF results.
4. `state.intent_hits` chứa intent rescue candidates riêng.
5. BGE score vẫn chạy trên global RRF pool.
6. Keep set mới chỉ được tạo khi analyze/package:

```python
keep = top80_bge | top80_rrf | intent_hits
```

7. Mục tiêu đánh giá là `Recall_keep`, không phải MRR hay Recall@40.
