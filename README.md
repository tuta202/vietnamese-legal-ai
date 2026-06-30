# Vietnamese Legal RAG System

Hướng dẫn này mô tả cách cài đặt, cấu hình và chạy lại toàn bộ hệ thống bằng
GPU backend để tạo `results.json` và `submission.zip`.

Các model của workflow:

- `Qwen3-Embedding-8B`: document, query, topic và intent embedding;
- `Gemma 3 12B-it`: query rewrite, intent decomposition, article verification và answer generation;
- `BAAI/bge-reranker-v2-m3`: intent-wise cross-encoder reranking.

Tài liệu chi tiết:

- [Runbook](legal-ai/docs/runbook.md)
- [RAG workflow](legal-ai/docs/rag-workflow.md)
- [Source layout](legal-ai/docs/source-layout.md)

## 1. Workflow

Pipeline xử lý article-level theo thứ tự:

1. Gemma rewrite câu hỏi.
2. Gemma decompose các legal intent độc lập.
3. Global retrieval bằng BM25 và dense embedding, sau đó RRF fusion.
4. Retrieval BM25 + dense riêng cho từng intent.
5. BGE rerank candidate theo từng intent và tạo tiered union.
6. Stage 1 compact verifier ưu tiên recall.
7. Penalty-aware deterministic cleanup.
8. Final collective compact verifier ưu tiên precision.
9. Enforcement-role gate.
10. Gemma sinh câu trả lời grounded từ các article cuối cùng.
11. Đóng gói `results.json` và `submission.zip`.

Request lỗi hoặc output không parse được không bị thay bằng fallback kỹ thuật.
Question lỗi được để lại và chạy lại bằng cơ chế resume.

## 2. Cấu trúc mã nguồn

```text
legal-ai/
|-- legal_rag/
|   |-- common/          shared utilities, retry, article lookup
|   |-- backends/        GPU endpoint adapters
|   |-- indexing/        BM25 và Qdrant indexing
|   |-- retrieval/       query analysis và retrieval
|   |-- ranking/         BGE rerank và tiered union
|   |-- verification/    LLM verifiers, cleanup và gates
|   |-- generation/      grounded answer generation
|   |-- output/          validation và submission packaging
|   `-- orchestration/   resumable end-to-end pipeline
|-- corpus/
|   `-- data/
|       `-- corpus_clean_asof_20260301.json
|-- retrieval/data/      BM25 indexes
|-- outputs/             run caches, logs, artifacts và submissions
|-- tests/
|-- config_gpu_clean.yaml
|-- build_core_bm25.py
|-- setup_qdrant_cloud.py
`-- run_best_pipeline.py
```

## 3. Môi trường

Môi trường tham chiếu:

- Windows PowerShell;
- Python 3.11;
- Google Cloud CLI để xác thực các GPU endpoint hiện tại;
- Qdrant Cloud hoặc Qdrant server tương thích API;
- ba model GPU đã deploy và truy cập được qua endpoint.

Cài dependency:

```powershell
cd C:\development\vietnamese-legal-ai\legal-ai

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Dữ liệu

Corpus:

```text
legal-ai/corpus/data/corpus_clean_asof_20260301.json
```

Tập câu hỏi mẫu trong các command:

```text
R2AIStage1DATA.json
```

## 5. Model và endpoint

Repo không đóng gói checkpoint. Cần deploy trước:

| Vai trò | Model |
|---|---|
| Embedding | `Qwen3-Embedding-8B` |
| LLM | `google/gemma-3-12b-it` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

`Qwen3-Embedding-8B` phải trả vector 4096 chiều. Endpoint LLM phải hỗ trợ
chat completion; endpoint embedding và BGE phải tương thích adapter trong
`legal_rag/backends/gpu.py`.

## 6. Biến môi trường

Tạo `.env` từ template:

```powershell
Copy-Item .env.example .env
```

Các biến bắt buộc:

```dotenv
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>

GCP_PROJECT=<your-project-id>

GPU_EMBED_ENDPOINT_ID=<qwen-embedding-endpoint-id>
GPU_EMBED_DNS=<qwen-embedding-dedicated-domain>

GPU_LLM_ENDPOINT_ID=<gemma-endpoint-id>
GPU_LLM_DNS=<gemma-dedicated-domain>

GPU_BGE_ENDPOINT_ID=<bge-endpoint-id>
GPU_BGE_DNS=<bge-dedicated-domain>
```

Xác thực:

```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project <GCP_PROJECT>
```

## 7. Cấu hình GPU

Pipeline dùng duy nhất:

```text
legal-ai/config_gpu_clean.yaml
```

Kiểm tra các mục trước khi chạy:

- `backend: gpu`;
- `qdrant.collection`;
- `qdrant.vector_size: 4096`;
- `gpu.embed_model: Qwen3-Embedding-8B`;
- `gpu.llm_model: google/gemma-3-12b-it`;
- `bge.*`;
- `retrieval.*`;
- `generator.*`.

Collection mặc định:

```text
legal_vn_qwen3_asof_20260301_v1
```

## 8. Build BM25

Chạy trong thư mục `legal-ai`:

```powershell
$env:PYTHONIOENCODING="utf-8"

.\.venv\Scripts\python.exe build_core_bm25.py `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --output retrieval\data\bm25_index_asof_20260301.pkl `
  --expected-count 82570
```

Output:

```text
retrieval/data/bm25_index_asof_20260301.pkl
```

## 9. Build Qdrant collection

Kiểm tra endpoint và Qdrant trước:

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --test-only
```

Build collection mới từ đầu:

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --workers 10 `
  --max-resume-passes 5 `
  --force
```

`--force` xóa và tạo lại đúng collection khai báo trong config. Chỉ sử dụng khi
chắc chắn collection name không trỏ vào dữ liệu cần giữ.

Resume build sau khi dừng:

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --workers 10 `
  --max-resume-passes 5
```

Khi resume, không truyền `--force`. Script kiểm tra metadata/hash của point,
retry batch lỗi và chỉ hoàn thành khi count cùng toàn bộ expected IDs hợp lệ.

## 10. Preflight

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py `
  --config config_gpu_clean.yaml `
  --input ..\R2AIStage1DATA.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --bm25-index retrieval\data\bm25_index_asof_20260301.pkl `
  --run-dir outputs\runs\gpu_asof_20260301 `
  --preflight-only
```

Preflight xác nhận input IDs, corpus, BM25, Qdrant collection, vector schema,
embedding fingerprint, config và endpoint trước khi xử lý 2.000 câu.

## 11. Chạy full pipeline

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py `
  --config config_gpu_clean.yaml `
  --input ..\R2AIStage1DATA.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --bm25-index retrieval\data\bm25_index_asof_20260301.pkl `
  --run-dir outputs\runs\gpu_asof_20260301 `
  --analysis-workers 12 `
  --retrieval-workers 20 `
  --bge-workers 12 `
  --llm-workers 12 `
  --max-resume-passes 5
```

Pipeline chạy hết 2.000 question của một phase trước khi chuyển sang phase kế
tiếp. Mỗi loại workload có worker riêng để phù hợp endpoint tương ứng.

## 12. Resume pipeline

Để resume, chạy lại chính xác command full với cùng:

- `--run-dir`;
- `--input`;
- `--corpus`;
- `--bm25-index`;
- `--config`.

Nếu chỉ endpoint ID/DNS thay đổi sau redeploy, thêm:

```text
--accept-runtime-change
```

Nếu chủ động resume sau khi sửa code hoặc workflow:

```text
--accept-code-change --accept-workflow-change
```

Chỉ dùng các flag chấp nhận thay đổi khi hiểu rõ thay đổi đó. Fingerprint vẫn
không cho phép âm thầm thay input, corpus hoặc BM25 của một run đang có cache.

## 13. Chạy từng phase

Liệt kê phase:

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py --list-stages
```

Ví dụ chạy đến Global RRF:

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py `
  --config config_gpu_clean.yaml `
  --input ..\R2AIStage1DATA.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --bm25-index retrieval\data\bm25_index_asof_20260301.pkl `
  --run-dir outputs\runs\gpu_phase_benchmark `
  --analysis-workers 12 `
  --retrieval-workers 20 `
  --bge-workers 12 `
  --llm-workers 12 `
  --max-resume-passes 5 `
  --stop-after-stage 02_global_rrf
```

Giữ nguyên command và `--run-dir`, sau đó đổi `--stop-after-stage` để chạy tiếp.
Mỗi phase article-selection tạo submission trung gian để benchmark leaderboard.

## 14. Kết quả

Một run có cấu trúc:

```text
outputs/runs/<run-name>/
|-- manifest.json
|-- errors.jsonl
|-- cache/
|-- artifacts/
|-- logs/
`-- submissions/
```

Kết quả cuối:

```text
outputs/runs/<run-name>/submissions/final_answers/results.json
outputs/runs/<run-name>/submissions/final_answers/submission.zip
```

## 15. Checklist tái hiện

1. Python 3.11 và `requirements.txt` đã được cài.
2. Ba GPU endpoint hoạt động và `.env` chứa đúng ID/DNS.
3. Application Default Credentials còn hiệu lực.
4. Corpus clean tồn tại và có đúng số article dự kiến.
5. BM25 được build từ chính corpus đó.
6. Qdrant collection được build bằng Qwen3-Embedding-8B, dimension 4096.
7. `--preflight-only` thành công.
8. Full run dùng một `--run-dir` riêng.
9. Không đổi input/corpus/BM25/config giữa các lần resume.
10. Kiểm tra đủ question IDs và không có technical fallback trước khi nộp.
