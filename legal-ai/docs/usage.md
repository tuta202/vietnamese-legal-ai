# Vietnamese Legal AI — Hướng dẫn sử dụng

Hệ thống RAG (Retrieval-Augmented Generation) trả lời câu hỏi pháp luật tiếng Việt.
Hỗ trợ hai backend: **Vertex AI / Gemini** (không cần GPU) và **vLLM / Qwen** (cần GPU).

---

## Cấu trúc thư mục

```
legal-ai/
├── pipeline.py              # Entry point chính
├── config_vertex.yaml       # Config cho Vertex AI backend
├── retrieval/config.yaml    # Config cho vLLM backend (default)
├── corpus/
│   ├── parser.py            # Parser văn bản pháp luật
│   ├── builder.py           # Build corpus.json từ file .txt
│   └── data/
│       ├── raw/             # 13 file .txt nguồn (read-only)
│       └── corpus.json      # 1044 articles đã parse
├── eval/
│   └── data/eval_set.json   # 50 câu hỏi để đánh giá
├── setup_qdrant_cloud.py    # Bootstrap Qdrant Cloud (Vertex path)
├── submit.py                # Validate + đóng gói nộp bài
└── .env.example             # Template cho secrets
```

---

## Cài đặt

### 1. Clone và cài dependencies

```bash
# Vertex AI backend (không cần GPU — khuyến nghị)
pip install -r requirements_vertex.txt

# vLLM / Qwen backend (cần GPU)
pip install -r requirements_vllm.txt
```

### 2. Cấu hình secrets

```bash
cp .env.example .env
# Mở .env và điền vào:
```

```env
# Qdrant Cloud
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<your-api-key>

# GCP / Vertex AI — chọn một trong hai:
GCP_PROJECT=<your-project-id>   # Option A: Application Default Credentials
GOOGLE_API_KEY=                  # Option B: API key (bỏ trống nếu dùng Option A)
```

Nếu dùng Option A (ADC), chạy thêm:

```bash
gcloud auth application-default login
```

---

## Chạy pipeline

### Vertex AI backend (Gemini + Qdrant Cloud)

```bash
python pipeline.py \
    --config config_vertex.yaml \
    --input eval/data/eval_set.json \
    --output results.json
```

### vLLM / Qwen backend (local GPU)

```bash
# Khởi động vLLM server trước (cần GPU)
python pipeline.py \
    --input eval/data/eval_set.json \
    --output results.json
```

### Mock mode (không cần GPU, không cần API — để test)

```bash
python pipeline.py \
    --config config_vertex.yaml \
    --mock \
    --input eval/data/eval_set.json \
    --output results_mock.json
```

### Format file input

```json
[
  {"id": 1, "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào?"},
  {"id": 2, "question": "Thuế suất thuế GTGT phổ thông là bao nhiêu?"}
]
```

### Format file output

```json
[
  {
    "id": 1,
    "question": "...",
    "answer": "Theo Điều 5 Luật Hỗ trợ DNNVV...",
    "relevant_docs": ["80/2021/NĐ-CP|Nghị định Hỗ trợ DNNVV"],
    "relevant_articles": ["80/2021/NĐ-CP|...|Điều 5"]
  }
]
```

---

## Đánh giá kết quả

```bash
python eval/evaluator.py \
    --eval-set eval/data/eval_set.json \
    --predictions results.json
```

Output:

```
Questions evaluated : 50
Macro F2            : 0.6322
Macro Precision     : 0.3600
Macro Recall        : 0.8200
Answer Coverage     : 96%
```

> **F2 score** = metric chính (recall được trọng số 4×, do bỏ sót điều luật tệ hơn cite nhầm).

---

## Đóng gói nộp bài

```bash
python submit.py --input results.json
```

Tạo `submission/submission.zip` chứa `results.json` đã validate.

Chỉ validate (không tạo ZIP):

```bash
python submit.py --validate-only results.json
```

---

## Bootstrap Qdrant Cloud (chỉ cần làm 1 lần)

Dùng khi lần đầu setup hoặc cần rebuild collection:

```bash
# Test kết nối (không embed, không ghi gì)
python setup_qdrant_cloud.py --config config_vertex.yaml --test-only

# Tạo collection và embed toàn bộ corpus (1044 articles, ~5–10 phút)
python setup_qdrant_cloud.py \
    --config config_vertex.yaml \
    --corpus corpus/data/corpus.json

# Rebuild từ đầu (xóa collection cũ)
python setup_qdrant_cloud.py \
    --config config_vertex.yaml \
    --corpus corpus/data/corpus.json \
    --force
```

---

## Rebuild corpus (nếu sửa parser hoặc raw files)

```bash
python corpus/builder.py
```

Output: `corpus/data/corpus.json` — hiện tại 1044 articles từ 13 văn bản pháp luật.

---

## Chạy tests

```bash
python -m pytest -q          # 122 tests
python -m pytest -q -k mock  # chỉ chạy mock tests (offline)
```

---

## Luồng hoạt động

```
Câu hỏi
   │
   ▼
QueryRewriter  ──► tạo thêm 2 query biến thể (Gemini / Qwen)
   │
   ▼
HybridSearch   ──► Dense (Qdrant) + BM25, RRF fusion → top-20
   │
   ▼
Reranker       ──► LLM cross-encoder → top-3
   │
   ▼
Generator      ──► Gemini / Qwen + system prompt 9 quy tắc → câu trả lời
   │
   ▼
Output JSON    ──► answer + relevant_docs + relevant_articles
```

---

## Xem thêm

- [`docs/gcp-setup-guide.md`](gcp-setup-guide.md) — hướng dẫn tạo GCP project, cấp quyền
- [`docs/gcp-quick-reference.md`](gcp-quick-reference.md) — lệnh gcloud hay dùng
- [`docs/vertex_setup_guide.md`](vertex_setup_guide.md) — setup Vertex AI + Qdrant Cloud chi tiết
