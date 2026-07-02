# Runbook GPU Cho RAG Pipeline

Tài liệu này mô tả cách chạy lại hệ thống bằng GPU backend trên Windows PowerShell.
Tất cả command đều chạy từ thư mục `legal-ai`.

Workflow chính:

- `Qwen3-Embedding-8B` cho embedding.
- `Gemma 3 12B-it` cho query analysis, verification và generation.
- `BAAI/bge-reranker-v2-m3` cho intent-wise reranking.

## 1. Chuẩn Bị Python

```powershell
cd C:\development\vietnamese-legal-ai\legal-ai

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Đặt UTF-8 cho PowerShell:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## 2. Xác Thực Endpoint

```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project <GCP_PROJECT_ID>
```

Tài khoản phải có quyền gọi cả ba endpoint embedding, LLM và BGE.

## 3. Biến Môi Trường

```powershell
Copy-Item .env.example .env
```

Điền các giá trị sau trong `.env`:

```dotenv
QDRANT_URL=https://<cluster>.cloud.qdrant.io
QDRANT_API_KEY=<api-key>

GCP_PROJECT=<project-id>

GPU_EMBED_ENDPOINT_ID=<qwen-embedding-endpoint-id>
GPU_EMBED_DNS=<qwen-embedding-dedicated-domain>

GPU_LLM_ENDPOINT_ID=<gemma-endpoint-id>
GPU_LLM_DNS=<gemma-dedicated-domain>

GPU_BGE_ENDPOINT_ID=<bge-endpoint-id>
GPU_BGE_DNS=<bge-dedicated-domain>
```

Ghi chú:

- Nếu chạy Qdrant local, có thể để `QDRANT_API_KEY=` trống.
- Nếu endpoint không có dedicated domain, để biến DNS trống theo adapter đang dùng.
- Endpoint ID, DNS, project và region phải khớp `config_gpu_clean.yaml`.

## 4. Corpus Và Input

Corpus chuẩn:

```text
corpus/data/corpus_clean_asof_20260301.json
```

Input benchmark:

```text
..\R2AIStage1DATA.json
```

Yêu cầu:

- corpus phải là bản clean as-of;
- corpus có `82.570` articles;
- input có `2.000` question IDs duy nhất.

## 5. Build BM25

```powershell
.\.venv\Scripts\python.exe build_core_bm25.py `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --output retrieval\data\bm25_index_asof_20260301.pkl `
  --expected-count 82570
```

Output:

```text
retrieval/data/bm25_index_asof_20260301.pkl
```

## 6. Build Qdrant Collection

### 6.1 Test trước

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --test-only
```

### 6.2 Build mới từ đầu

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --workers 10 `
  --max-resume-passes 5 `
  --force
```

### 6.3 Resume sau khi dừng

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --workers 10 `
  --max-resume-passes 5
```

Ghi chú:

- `--force` chỉ dùng khi muốn rebuild collection.
- Khi resume, không dùng `--force`.
- Builder sẽ retry batch lỗi và xác minh lại schema, count và toàn bộ expected IDs.

## 7. Preflight

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py `
  --config config_gpu_clean.yaml `
  --input ..\R2AIStage1DATA.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --bm25-index retrieval\data\bm25_index_asof_20260301.pkl `
  --run-dir outputs\runs\gpu_asof_20260301 `
  --preflight-only
```

Preflight kiểm tra:

- input IDs;
- corpus và BM25;
- Qdrant collection, schema và embedding hash;
- endpoint embedding, rewrite, intent decomposition và BGE;
- fingerprint của input, corpus, BM25, config, runtime và code.

Preflight không chạy 2.000 câu.

## 8. Chạy Full Pipeline

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
  --rescue-coverage-depth 4 `
  --max-resume-passes 5
```

Nếu có question lỗi do network, GPU hoặc endpoint tạm thời, chạy lại đúng command
cũ với cùng `--run-dir`; pipeline sẽ tự bỏ qua question đã có cache hợp lệ và
tiếp tục phần còn thiếu.

## 9. Resume

Chạy lại đúng command full với cùng:

- `--run-dir`
- `--input`
- `--corpus`
- `--bm25-index`
- `--config`

Nếu có question lỗi do network, GPU hoặc endpoint tạm thời thì cũng chỉ cần
chạy lại đúng command cũ với cùng `--run-dir`. Không cần sửa cache thủ công;
pipeline sẽ resume phần chưa xong.

Nếu chỉ endpoint đổi sau redeploy, thêm:

---

```text
--accept-runtime-change
```

Nếu code hoặc workflow đổi, thêm:

```text
--accept-code-change --accept-workflow-change
```

## 10. Output

Run output nằm ở:

```text
outputs/runs/<run-name>/
```

Cấu trúc chính:

```text
cache/
artifacts/
logs/
submissions/
manifest.json
errors.jsonl
```

Submission cuối:

```text
outputs/runs/<run-name>/submissions/final_answers_rescue_depth4/results.json
outputs/runs/<run-name>/submissions/final_answers_rescue_depth4/submission.zip
```

## 11. Checklist Nhanh

1. `.env` đã có đủ Qdrant, project và ba GPU endpoint.
2. ADC còn hiệu lực.
3. Corpus và BM25 khớp nhau.
4. Qdrant collection đã build đúng corpus.
5. `--preflight-only` chạy qua.
6. Full run dùng một `--run-dir` riêng.
7. Khi lỗi thì resume bằng đúng command cũ.
