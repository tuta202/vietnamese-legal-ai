# Luồng Retrieval — Hybrid BM25 + Dense + RRF

Tài liệu mô tả chi tiết **giai đoạn truy hồi (retrieval)** của hệ thống Vietnamese
Legal AI: nó nhận gì, làm gì qua từng bước, và xuất ra gì cho tầng rerank.

> Phạm vi: đây là **Step 2** trong pipeline 5 bước
> `rewrite → **retrieve** → rerank → generate → format`.
> Code chính: `pipeline.py::step_retrieve`, `retrieval/bm25_index.py`,
> `retrieval/embedder.py`, `retrieval/query_rewriter.py`, `retrieval/hybrid_search.py`.
>
> **Cập nhật TIP-019:** rerank trở về **single-tier** whole-pool LLM (bỏ BGE
> two-tier vì làm giảm F2 — xem §9) và **bỏ query decomposition** (sub-queries
> làm regress). Tài liệu này đã được cập nhật cho code hiện tại: `step_retrieve`
> chỉ chạy **BM25(rewritten) + Dense(rewritten) + Dense(topic) → RRF**, pool
> `top_k_fusion=50`. Tham số theo `config_gpu.yaml`.

---

## 1. Tổng quan — "hybrid"

Một câu hỏi pháp lý có thể khớp tài liệu theo **hai cách khác nhau**, và không cách
nào đủ một mình:

| Kiểu khớp | Điểm mạnh | Điểm yếu |
|-----------|-----------|----------|
| **Lexical (BM25)** | Khớp đúng thuật ngữ luật, số hiệu, từ hiếm ("doanh nghiệp siêu nhỏ", "04/2017/QH14") | Mù nghĩa — không hiểu "sa thải" ≈ "chấm dứt hợp đồng" |
| **Dense (vector)** | Hiểu ngữ nghĩa, diễn đạt khác từ vẫn khớp | Hay "trôi" sang chủ đề gần giống; bỏ lỡ từ khóa hiếm |

Hệ thống chạy **cả hai song song** rồi hợp nhất bằng **Reciprocal Rank Fusion (RRF)**.
Đây là cơ chế **recall-first**: gom thật rộng (50 ứng viên) để tầng rerank phía sau
lọc tinh (50 → 6).

```
                          câu hỏi gốc
                              │
                        ┌─────▼─────┐
                        │  REWRITE  │  (Step 1 — Gemma)
                        └─────┬─────┘
              rewritten_query + topic_description
                              │
   ┌──────────────┬───────────┴──────────┐
   ▼              ▼                       ▼
┌───────┐   ┌───────────┐         ┌──────────────┐
│ BM25  │   │   DENSE    │        │    DENSE     │
│(rewrt)│   │  (rewrt)   │        │ (topic_desc) │
└───┬───┘   └─────┬──────┘        └──────┬───────┘
    │ ranked      │ ranked               │ ranked (nếu topic ≠ rewritten)
    └─────────────┴──────────┬───────────┘
                             ▼
                      ┌─────────────┐
                      │ RRF FUSION  │  (rrf_k = 60)
                      └──────┬──────┘
                             ▼ top_k_fusion = 50
                      ┌─────────────┐
                      │  RESOLVE    │  (gắn payload đầy đủ cho mỗi chunk)
                      │  PAYLOADS   │
                      └──────┬──────┘
                             ▼
                   50 ứng viên  →  tầng RERANK (single-tier LLM → 6)
```

---

## 2. Đầu vào: kết quả của Query Rewriting

Retrieval **không** dùng câu hỏi thô. Step 1 (`step_rewrite`) gọi LLM (Gemma trên
GPU) sinh ra **2 trường** — xem `retrieval/query_rewriter.py`:

- **`rewritten_query`** — truy vấn cô đọng, tập trung khái niệm pháp lý cốt lõi.
  Ví dụ: *"Công ty tôi có 5 người, có phải DNNVV không?"* →
  *"tiêu chí xác định doanh nghiệp nhỏ và vừa số lao động vốn doanh thu"*.
- **`topic_description`** — mô tả CHỦ ĐỀ + THUẬT NGỮ + loại văn bản liên quan
  (Luật/Nghị định/Thông tư), **cố ý không chứa số liệu cụ thể hay "Điều X"**.

> **Vì sao không dùng HyDE?** HyDE (sinh câu trả lời giả để embed) dễ bịa số hiệu
> luật / con số lỗi thời. `topic_description` neo embedding vào *không gian khái
> niệm* thay vì *facts bịa*, an toàn hơn cho domain pháp lý.

Nếu LLM lỗi mạng / parse hỏng → rewriter **fallback** trả về câu hỏi gốc cho
`rewritten_query` (không bao giờ chặn pipeline).

---

## 3. Nhánh A — BM25 (lexical search)

File: `retrieval/bm25_index.py`. Đây là **Okapi BM25** tự cài (inverted index),
không phụ thuộc dịch vụ ngoài → cực nhanh, chạy in-process. BM25 chạy cho
**truy vấn rewritten** (hoặc câu hỏi gốc nếu rewrite rỗng).

### 3.1 Tokenizer
`vietnamese_simple_tokenize()`: lowercase → bỏ dấu câu (`[^\w\s]`, giữ ký tự
tiếng Việt có dấu) → tách theo khoảng trắng → bỏ token độ dài ≤ 1.
*(Tokenize theo âm tiết, không tách từ ghép — đơn giản, đủ tốt cho BM25.)*

### 3.2 Index
Mỗi **điều luật** (article) được index trên `dieu_title + content`:
- `_inverted`: `term → [(doc_idx, term_freq), ...]`
- `_idf[term] = log((N - df + 0.5) / (df + 0.5) + 1)`
- `_avgdl`: độ dài tài liệu trung bình (chuẩn hóa độ dài).
- Tham số Okapi: `K1 = 1.5`, `B = 0.75`.

### 3.3 Công thức tính điểm
Với mỗi token trong truy vấn, cộng dồn vào mỗi tài liệu chứa token:

```
score += idf(token) · [ f·(K1+1) ] / [ f + K1·(1 − B + B·dl/avgdl) ]
```

(`f` = tần suất token trong tài liệu, `dl` = độ dài tài liệu). Trả về
`top_k_bm25 = 80` cặp `(chunk_id, score)` sắp xếp giảm dần.

Index được build offline (`python retrieval/bm25_index.py`) và lưu pickle
`retrieval/data/bm25_index.pkl`; runtime chỉ `load()`.

---

## 4. Nhánh B — Dense / semantic search

File: `retrieval/embedder.py` (+ override GPU trong `gpu_backends.py`),
truy vấn vector qua **Qdrant**. `step_retrieve` embed `rewritten_query` và
`topic_description` (nếu khác) bằng `embedder.embed_query(text)` cho **từng** truy vấn.

### 4.1 Embedding model — Qwen3-Embedding-8B (4096 chiều)
Model **bất đối xứng (asymmetric)** giữa query và document:

- **Query** được bọc tiền tố hướng dẫn (instruction-prefixed) — hằng số
  `_EMBED_INSTRUCTION` **hardcode** trong `embedder.py::_format_query`:
  ```
  Instruct: {_EMBED_INSTRUCTION}
  Query: {text}
  ```
- **Document** được embed **THÔ** (chỉ `law_id | law_type law_name | dieu_number`
  + title + 512 ký tự đầu content), **không** có tiền tố Instruct.

> **Hệ quả vận hành quan trọng:** instruction chỉ áp lên vector *query* lúc chạy →
> đổi nó **không cần re-embed lại ~113k điều** trong corpus.

Vector luôn được **L2-normalize** (cosine similarity ⇔ dot product).

### 4.2 Hai backend embed (cùng một interface)
- **GPU** (`GpuEmbedder._embed`): gọi endpoint Vertex AI **dedicated** qua route
  TEI `:predict` (KHÔNG phải OpenAI):
  ```
  POST https://{embed_dns}/v1/projects/{proj}/locations/{embed_region}/endpoints/{id}:predict
  body  {"instances": [{"inputs": "<text>"}, ...]}
  resp  {"predictions": [[[...4096 floats...]], ...]}
  ```
  Embed nằm ở **region riêng** (`asia-northeast1`), khác region LLM
  (`asia-southeast1`) → config tách `embed_region`/`embed_dns`. **DNS dedicated xoay
  vòng mỗi lần redeploy** → phải refresh `GPU_EMBED_DNS` trong `.env`.

### 4.3 Truy vấn Qdrant
`pipeline.py::_dense_search` gọi `qdrant.query_points(collection, query=vec,
limit=top_k_dense=80, with_payload=True)` cho **mỗi vector**, có **retry backoff**
(5 lần, 1→2→4…s) để chịu được Qdrant chớp tắt. Trả về `(hits, payloads)`:
- `hits = [(chunk_id, score), ...]` — **keyed bằng `chunk_id` thô** lấy từ payload
  (không phải UUID điểm), để khớp với key của BM25 khi fusion.
- `payloads = {chunk_id: payload}` — để dựng lại điều cho hit chỉ có ở nhánh dense
  (gom dồn từ mọi vector vào một dict chung).

Collection: `legal_vn_garden` (4096d, ~113.5k điểm, Qdrant local Docker
`localhost:6333`).

---

## 5. Các ranked list đi vào fusion

`step_retrieve` tạo ra **2–3 ranked list**:

| Nguồn | Số list |
|-------|---------|
| BM25(`rewritten_query`) | 1 |
| Dense(`rewritten_query`) | 1 |
| Dense(`topic_description`) *(nếu khác rewritten)* | 0–1 |

→ Tối thiểu 2 list (khi `topic_description` trùng/rỗng), tối đa 3. RRF **thưởng**
điều xuất hiện nhất quán ở cả nhánh lexical lẫn semantic → recall ổn định hơn.

---

## 6. Hợp nhất — Reciprocal Rank Fusion (RRF)

File: `retrieval/hybrid_search.py::rrf_fusion`.

```python
score[chunk_id] = Σ_lists  1 / (k + rank_trong_list)      # k = rrf_k = 60
```

Đặc điểm:
- **Dựa trên THỨ HẠNG, không dựa trên điểm thô** → không cần chuẩn hóa thang điểm
  giữa BM25 (không chặn trên) và cosine (0–1). Đây là điểm mấu chốt khiến RRF bền
  (không list nào "áp đảo" bằng scale).
- Hằng số `k = 60` làm phẳng ảnh hưởng của các rank đầu (giảm thống trị của top-1).
- Một chunk có mặt ở nhiều list → cộng dồn nhiều số hạng → đẩy lên cao.

Sau fusion, cắt lấy **`top_k_fusion = 50`** ứng viên hàng đầu.

---

## 7. Resolve payloads — gắn nội dung điều

`fused` mới chỉ là `(chunk_id, rrf_score)`. Bước resolve gắn **payload đầy đủ**
(law_id, law_name, dieu_number, dieu_title, content, `relevant_doc_str`,
`relevant_article_str`) cho từng chunk:

1. Tra trong **BM25 payload** trước (`get_payload`) — đa số hit có sẵn.
2. Hit chỉ có ở nhánh dense → lấy từ **`payloads` của Qdrant** đã gom ở §4.3
   (chuẩn hóa qua `_normalize_payload`).
3. Không thấy đâu cả → giữ tối thiểu `{"chunk_id": ...}` (hiếm).

Kết quả: `state.fused_results` = list **≤ 50** dict, mỗi dict kèm `rrf_score`.

### Fallback an toàn
Nếu danh sách rỗng (vd `rewritten_query` lệch hẳn) → chạy lại **BM25 trên câu hỏi
GỐC** với `top_k_fusion`, đảm bảo **không bao giờ** đưa context rỗng xuống generate.

---

## 8. Các tham số điều chỉnh (config `retrieval:`)

Từ `config_gpu.yaml` (đường chạy hiện tại):

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `top_k_bm25` | 80 | số hit BM25 lấy ra |
| `top_k_dense` | 80 | `limit` mỗi lượt query Qdrant |
| `top_k_fusion` | **50** | kích thước pool sau RRF → đầu vào rerank |
| `top_k_rerank` | 6 | số điều trích dẫn cuối (single-tier LLM) |
| `rrf_k` | 60 | hằng số RRF |
| `embedding.dimension` | 4096 | = `qdrant.vector_size` (bắt buộc khớp) |

> **Bất biến bắt buộc:** `embedding.dimension == qdrant.vector_size` (lệch →
> validate fail). Phễu phải đơn điệu: `top_k_rerank ≤ top_k_fusion`.

> **Hai config song song:** `config_gpu.yaml` (Qwen3 + Gemma trên endpoint GPU) và
> `config_vertex.yaml` (Gemini). Hai config **hoạt động giống hệt nhau** — cùng
> tham số retrieval/rerank/generate, **chỉ khác model** (và `dimension`/collection
> đi kèm). Không còn tầng BGE two-tier hay decomposition.

---

## 9. Bàn giao cho tầng Rerank

`state.fused_results` (≤ 50 ứng viên) là **đầu ra của retrieval**. Tầng sau là
**single-tier** whole-pool LLM rerank:

```
retrieval 50  ──►  LLM whole-pool rerank  ──►  6 (cited)
                   (1 call, thang điểm 0–10 chung)     (top_k_rerank)
```

> **Lịch sử:** TIP-013/014 từng dùng **two-tier** `80 → BGE cross-encoder → 15 →
> CoT → 5`. TIP-016 sweep ngưỡng cho thấy **mọi biến thể two-tier đều thua**
> baseline single-tier (F2≈0.5033) → revert về single-tier. TIP-017 thêm query
> decomposition nhưng cũng regress → TIP-019 **bỏ luôn decomposition**. Code
> two-tier và decomposition đã được gỡ hẳn.

Retrieval **không** quyết định điều nào được trích dẫn — nó chỉ đảm bảo **recall**:
"điều đúng phải nằm đâu đó trong 50". Tầng rerank (cùng prompt + cùng cách chấm
0–10 cho cả backend gpu lẫn vertex) mới chọn ra điều trích dẫn cuối.

---

## 10. Bản đồ code (tham chiếu nhanh)

| Chức năng | Vị trí |
|-----------|--------|
| Orchestrate retrieval (BM25 + dense + RRF) | `pipeline.py::step_retrieve` |
| Query rewriting (rewritten_query + topic_description) | `retrieval/query_rewriter.py` |
| Dense search + retry + payload keying | `pipeline.py::_dense_search` |
| RRF | `retrieval/hybrid_search.py::rrf_fusion` |
| BM25 (index + Okapi) | `retrieval/bm25_index.py` |
| Embedder (TEI :predict / Gemini) | `gpu_backends.py::GpuEmbedder` / `vertex_backends.py::VertexEmbedder` |
| Rerank + score-saving (chung 2 backend) | `retrieval/reranker.py::rerank_with_scores` |

> Ghi chú kiến trúc: `pipeline.py::step_retrieve` là đường chạy production; mỗi step
> chạy/test độc lập, backend (gpu | vertex_ai) thay model bằng cách override hook.
> `HybridSearcher` trong `hybrid_search.py` là bản đóng gói độc lập (mock được để
> test offline) — không dùng trong production.
