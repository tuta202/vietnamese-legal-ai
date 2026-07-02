# Vietnamese Legal RAG System

Hướng dẫn này mô tả cách cài đặt, cấu hình và chạy lại toàn bộ hệ thống bằng
GPU backend để tạo `results.json` và `submission.zip`.

Các model của workflow:

- `Qwen3-Embedding-8B`: document, query, topic và intent embedding;
- `Gemma 3 12B-it`: query rewrite, intent decomposition, article verification và answer generation;
- `BAAI/bge-reranker-v2-m3`: intent-wise cross-encoder reranking.

Tài liệu chi tiết:

- [RAG workflow](legal-ai/docs/rag-workflow.md)
- [Source layout](legal-ai/docs/source-layout.md)
- [Runbook](legal-ai/docs/runbook.md)

Tài liệu Google Docs riêng của bạn giữ vai trò hướng dẫn vận hành endpoint và điền biến môi trường: [Google Docs](https://docs.google.com/document/d/1FYd9rO_qy03swiO8t4tKvpDRzpNqe9CbGl67_Th_-JQ/edit?usp=sharing).

## 1. Workflow

Pipeline xử lý theo thứ tự:

1. Gemma rewrite câu hỏi.
2. Gemma decompose các legal intent độc lập.
3. Global retrieval bằng BM25 và dense embedding, sau đó RRF fusion.
4. Retrieval BM25 + dense riêng cho từng intent.
5. BGE rerank candidate theo từng intent và tạo tiered union.
6. Stage 1 compact verifier ưu tiên recall.
7. Penalty-aware deterministic cleanup.
8. Final collective compact verifier ưu tiên precision.
9. Enforcement-role gate.
10. Raw-intent top1 coverage rescue để bổ sung căn cứ còn thiếu theo legal intent.
11. Gemma sinh câu trả lời grounded từ các article cuối cùng.
12. Đóng gói `results.json` và `submission.zip`.

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
- Python 3.14.3;
- Google Cloud CLI để xác thực các GPU endpoint hiện tại;
- Qdrant Cloud hoặc Qdrant server tương thích API;
- ba model GPU đã deploy và truy cập được qua endpoint.

Cài dependency:

```powershell
cd C:\development\vietnamese-legal-ai\legal-ai

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Dữ liệu

Dùng bộ dữ liệu đã chuẩn bị sẵn để đỡ tốn thời gian build lại BM25 + Qdrant Collection,
hãy tải tại:

[Google Drive bundle](https://drive.google.com/drive/folders/1RhGcsvwtEGA3_wCehafKeV0Om_4dRLUX?usp=drive_link)

Gói dữ liệu đã bao gồm:

- `corpus_clean_asof_20260301.json`: corpus pháp luật đã làm sạch và lọc hiệu lực tại ngày `01/03/2026`.
- `bm25_index_asof_20260301.pkl`: BM25 index đã được build sẵn từ corpus trên.
- Qdrant snapshot: snapshot của vector collection đã được build từ cùng corpus.

Sau khi tải về, đặt file vào đúng thư mục được dùng trong command:

- corpus: `legal-ai/corpus/data/corpus_clean_asof_20260301.json`
- BM25: `legal-ai/retrieval/data/bm25_index_asof_20260301.pkl`
- Qdrant snapshot: vào Qdrant local để upload snapshot và đặt đúng tên collection như trong `config_gpu_clean.yaml`

Nếu đã có đủ ba file này thì không cần build lại corpus, BM25 hay Qdrant collection từ đầu.

Corpus:

```text
legal-ai/corpus/data/corpus_clean_asof_20260301.json
```

Tập câu hỏi mẫu trong các command:

```text
R2AIStage1DATA.json
```

## 6. Biến môi trường

Tạo `.env` từ template:

```powershell
Copy-Item .env.example .env
```

Các biến bắt buộc:

```dotenv
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>
```

Nếu chạy Qdrant local (`http://localhost:6333`), có thể để `QDRANT_API_KEY` trống.

```dotenv
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

Chỉ dùng các flag chấp nhận thay đổi khi hiểu rõ thay đổi đó. Fingerprint vẫn
không cho phép âm thầm thay input, corpus hoặc BM25 của một run đang có cache.

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
outputs/runs/<run-name>/submissions/final_answers_rescue_depth4/results.json
outputs/runs/<run-name>/submissions/final_answers_rescue_depth4/submission.zip
```

## 15. Checklist tái hiện

1. Python 3.14.3 và `requirements.txt` đã được cài.
2. Ba GPU endpoint hoạt động và `.env` chứa đúng ID/DNS.
3. Application Default Credentials còn hiệu lực.
4. Corpus clean tồn tại và có đúng số article dự kiến.
5. File `.pkl` BM25 được build từ đúng corpus.
6. Qdrant collection được build bằng Qwen3-Embedding-8B, dimension 4096.
7. `--preflight-only` thành công.
8. Full run dùng một `--run-dir` riêng.
