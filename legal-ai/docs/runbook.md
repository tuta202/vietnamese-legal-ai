# Runbook GPU Cho RAG Pipeline

Tài liệu này hướng dẫn vận hành workflow hiện tại trên Windows PowerShell. Tất
cả command được chạy từ thư mục `legal-ai`.

Workflow chỉ sử dụng:

- `Qwen3-Embedding-8B` cho embedding;
- `Gemma 3 12B-it` cho query analysis, verification và generation;
- `BAAI/bge-reranker-v2-m3` cho intent-wise reranking.

## 1. Chuẩn Bị Python

```powershell
cd C:\development\vietnamese-legal-ai\legal-ai

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nên đặt UTF-8 trước khi chạy để PowerShell hiển thị tiếng Việt chính xác:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## 2. Xác Thực Endpoint

Các GPU endpoint hiện tại sử dụng Application Default Credentials:

```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project <GCP_PROJECT_ID>
```

Tài khoản hoặc service account phải có quyền gọi cả ba endpoint embedding, LLM
và BGE trong project/region được khai báo ở config.

## 3. Biến Môi Trường

Tạo `.env`:

```powershell
Copy-Item .env.example .env
```

Các giá trị cần điền:

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

Lưu ý:

- `.env` đã được gitignore; không commit secret.
- Nếu endpoint không có dedicated domain, để biến DNS trống nếu adapter hỗ trợ
  shared endpoint tương ứng.
- Endpoint ID, DNS, project và region phải khớp `config_gpu_clean.yaml`.
- Nếu dùng Qdrant local, đặt `QDRANT_URL=http://localhost:6333` và có thể để API
  key trống.

## 4. Corpus Chuẩn

Workflow hiện dùng duy nhất:

```text
corpus/data/corpus_clean_asof_20260301.json
```

Đây là corpus đã clean và lọc theo mốc hiệu lực `2026-03-01`:

- `82.570` articles;
- mỗi article có `chunk_id` duy nhất;
- mỗi article có `relevant_article_str`;
- metadata và Unicode đã được chuẩn hóa.

Cùng file này phải được dùng để build BM25, build Qdrant và hydrate nội dung
article. Không trộn index hoặc collection được tạo từ corpus khác.

Input benchmark mặc định:

```text
..\R2AIStage1DATA.json
```

Input này phải có `2.000` question IDs duy nhất.

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

Builder dừng nếu count hoặc dữ liệu corpus không hợp lệ. Pipeline preflight sẽ
đối chiếu lại BM25 IDs với corpus trước khi retrieval.

## 6. Build Qdrant Collection

Thông số hiện tại:

| Config | Embedder | Dimension | Collection mặc định |
|---|---|---:|---|
| `config_gpu_clean.yaml` | `Qwen3-Embedding-8B` | 4096 | `legal_vn_qwen3_asof_20260301_v1` |

### 6.1. Kiểm tra trước khi index

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --test-only
```

### 6.2. Tạo collection mới từ đầu

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --workers 10 `
  --max-resume-passes 5 `
  --force
```

`--force` xóa và tạo lại đúng collection trong config. Chỉ dùng khi chắc chắn
collection name không trỏ tới dữ liệu cần giữ.

### 6.3. Resume collection đang build

Khi process bị dừng, chạy lại nhưng bỏ `--force`:

```powershell
.\.venv\Scripts\python.exe setup_qdrant_cloud.py `
  --config config_gpu_clean.yaml `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --workers 10 `
  --max-resume-passes 5
```

Builder sẽ:

1. đọc các point hiện có;
2. chỉ reuse point có đúng backend, model và embedding-text hash;
3. retry batch embedding/upsert bị lỗi;
4. chạy lại các pass còn thiếu;
5. xác minh schema, exact count và toàn bộ expected IDs trước khi thành công.

Không resume bằng `--force`, vì collection đang build sẽ bị xóa.

## 7. Preflight Pipeline

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

- `2.000` question IDs;
- `82.570` corpus articles;
- BM25 IDs khớp corpus;
- Qdrant schema, count, IDs và embedding hash;
- dense search thực tế;
- LLM query analysis endpoint;
- BGE endpoint;
- fingerprint input, corpus, BM25, config, runtime và code.

Preflight không chạy pipeline 2.000 câu.

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

Lệnh trên chạy bản production depth 4 từ đầu đến Answer Generation. Để chạy
biến thể depth 2 trong một run độc lập, dùng `--run-dir` khác để cache và output
không ghi đè nhau:

```powershell
.\.venv\Scripts\python.exe run_best_pipeline.py `
  --config config_gpu_clean.yaml `
  --input ..\R2AIStage1DATA.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --bm25-index retrieval\data\bm25_index_asof_20260301.pkl `
  --run-dir outputs\runs\gpu_asof_20260301_depth2 `
  --analysis-workers 12 `
  --retrieval-workers 20 `
  --bge-workers 12 `
  --llm-workers 12 `
  --rescue-coverage-depth 2 `
  --max-resume-passes 5
```

Chạy lại nguyên lệnh với cùng `--run-dir` để resume. Không thêm `--force` và
không đổi depth giữa chừng trong cùng một run.

Pipeline xử lý xong toàn bộ question của một phase rồi mới chuyển sang phase kế
tiếp. Worker được tách theo loại workload:

- `--analysis-workers`: query rewrite và intent decomposition;
- `--retrieval-workers`: embedding query/topic/intent và retrieval;
- `--bge-workers`: BGE reranking;
- `--llm-workers`: Stage 1, Final verifier và generation.

## 9. Các Phase Hiện Tại

```text
01_query_analysis
02_global_rrf
03_raw_intent_retrieval
04_bge_intent_rerank
05_tiered_union
06_stage1_compact
07_penalty_cleanup
08_final_collective
09_enforcement_role_gate
10_intent_coverage_rescue
11_answer_generation
```

Chi tiết:

| Phase | Chức năng | Output có thể nộp |
|---|---|---|
| `01_query_analysis` | Hai LLM call: rewrite và intent decomposition | Không |
| `02_global_rrf` | BM25 + dense query/topic, RRF top 60 | `submissions/rrf_top60_clean/submission.zip` |
| `03_raw_intent_retrieval` | BM25 + dense + RRF cho từng intent | `submissions/raw_intent_top5_union/submission.zip` |
| `04_bge_intent_rerank` | BGE rerank keep theo intent | `submissions/bge_intent_top5_union/submission.zip` |
| `05_tiered_union` | Union RRF12 + BGE5/intent + raw-intent5/intent | `submissions/tiered_rrf12_bge5_rawintent5/submission.zip` |
| `06_stage1_compact` | Gemma recall-oriented, compact, batch 6, content 1.800 | `submissions/stage1_gemma_compact_v4/submission.zip` |
| `07_penalty_cleanup` | Conservative penalty-aware cleanup | `submissions/stage1_gemma_v4_penalty_cleanup/submission.zip` |
| `08_final_collective` | Gemma precision-oriented, compact, content 2.200 | `submissions/final_collective/submission.zip` |
| `09_enforcement_role_gate` | Loại điều xử phạt/cưỡng chế sai vai trò | `submissions/best_final_enforcement_gate/submission.zip` |
| `10_intent_coverage_rescue` | Rescue raw-intent top1; depth 4 mặc định | `submissions/best_final_enforcement_gate_rawintent_top1_rescue_depth4/submission.zip` |
| `11_answer_generation` | Sinh answer grounded từ article sau rescue | `submissions/final_answers_rescue_depth4/submission.zip` |

Bản leaderboard tốt nhất hiện tại của run GPU benchmark là post-process thử nghiệm sau phase 09:

```text
submissions/best_final_enforcement_gate_rawintent_top1_rescue_depth4/submission.zip
```

Depth 4 là bản production/best tổng thể:

- `ARTICLES_PRECISION=0.4805`, `ARTICLES_RECALL=0.7510`, `ARTICLES_F2MACRO=0.6411`;
- `DOCS_PRECISION=0.5367`, `DOCS_RECALL=0.7867`, `DOCS_F2MACRO=0.6928`;
- mean `3.6515` article/câu.

Bản thứ hai ưu tiên article recall là:

```text
submissions/best_final_enforcement_gate_rawintent_top1_rescue_depth2/submission.zip
```

Depth 2 đạt `ARTICLES_PRECISION=0.4794`, `ARTICLES_RECALL=0.7577`,
`ARTICLES_F2MACRO=0.6425`; mean `3.7485` article/câu. Hai cấu hình rescue raw-intent top1
khi final không còn coverage trong raw-intent top 4 hoặc top 2 tương ứng. Đây là phase 10
chính thức của `run_best_pipeline.py`; depth 4 là mặc định.

Prompt hiện tại:

- Stage 1: `gemma-recall-v4`;
- Final collective: `gemma-precision-v5`;
- Final không preserve top1 vô điều kiện;
- lỗi request/parse trong strict mode không được cache thành fallback.

## 10. Chạy Đến Một Phase

### Chạy riêng Answer Generation từ kết quả rescue hiện có

Depth 4:

```powershell
.\.venv\Scripts\python.exe -m legal_rag.generation.generate_answers `
  --config config_gpu_clean.yaml `
  --input outputs\runs\gpu_phase_benchmark\submissions\best_final_enforcement_gate_rawintent_top1_rescue_depth4\results.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --cache outputs\runs\gpu_phase_benchmark\cache\answer_generation_rescue_depth4.jsonl `
  --output-dir outputs\runs\gpu_phase_benchmark\submissions\final_answers_rescue_depth4 `
  --errors outputs\runs\gpu_phase_benchmark\artifacts\answer_generation_depth4_errors.json `
  --workers 12 `
  --resume `
  --strict-errors
```

Depth 2:

```powershell
.\.venv\Scripts\python.exe -m legal_rag.generation.generate_answers `
  --config config_gpu_clean.yaml `
  --input outputs\runs\gpu_phase_benchmark\submissions\best_final_enforcement_gate_rawintent_top1_rescue_depth2\results.json `
  --corpus corpus\data\corpus_clean_asof_20260301.json `
  --cache outputs\runs\gpu_phase_benchmark\cache\answer_generation_rescue_depth2.jsonl `
  --output-dir outputs\runs\gpu_phase_benchmark\submissions\final_answers_rescue_depth2 `
  --errors outputs\runs\gpu_phase_benchmark\artifacts\answer_generation_depth2_errors.json `
  --workers 12 `
  --resume `
  --strict-errors
```

Nếu request gặp lỗi kỹ thuật, chạy lại đúng lệnh tương ứng. Những câu đã có
cache hợp lệ sẽ được bỏ qua; câu lỗi hoặc chưa chạy sẽ được xử lý lại.

Liệt kê phase hợp lệ:

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

Muốn chạy tiếp đến phase 07, giữ nguyên mọi tham số và đổi dòng cuối:

```powershell
--stop-after-stage 07_penalty_cleanup
```

Các phase đã hoàn thành sẽ được validate và bỏ qua. Pipeline chỉ chạy phase còn
thiếu theo đúng thứ tự; không thể nhảy qua dependency chưa hoàn thành.

## 11. Resume Sau Khi Lỗi Hoặc Dừng Process

Chạy lại chính command cũ với cùng:

- `--run-dir`;
- `--input`;
- `--corpus`;
- `--bm25-index`;
- `--config`;
- workflow settings.

Mỗi resumable phase chỉ chạy question chưa có cache hợp lệ. Request, network hoặc
JSON parse lỗi không được ghi thành kết quả giả. Nếu vẫn còn lỗi sau số pass cho
phép, pipeline dừng với exit code khác `0` và không coi stage là hoàn thành.

### Endpoint vừa được redeploy

Nếu chỉ endpoint ID/DNS thay đổi:

```text
--accept-runtime-change
```

### Code hoặc workflow chủ động thay đổi

```text
--accept-code-change --accept-workflow-change
```

Nếu endpoint cũng đổi, dùng cả ba flag:

```text
--accept-code-change --accept-workflow-change --accept-runtime-change
```

Các flag này không cho phép âm thầm thay input, corpus hoặc BM25 của run cũ.

Workflow vừa chuyển sang Stage 1 v4 và Final v5. Khi resume một run được tạo bởi
prompt/cache version cũ, cần `--accept-code-change --accept-workflow-change`.
Orchestrator sử dụng cache path mới nên các phase verifier sẽ chạy lại, còn
artifact upstream hợp lệ có thể được reuse.

## 12. Cache Và Chính Sách Lỗi

Các cache chính trong `--run-dir/cache`:

```text
rewrite.jsonl
intents.jsonl
intent_ranked_hits.jsonl
bge_intent_scores.jsonl
stage1_gemma_compact_v4.jsonl
final_collective_gemma_v5.jsonl
answer_generation_rescue_depth4.jsonl
# hoặc answer_generation_rescue_depth2.jsonl
```

Nguyên tắc:

- chỉ append record đã hoàn thành;
- technical error không tạo fallback record trong strict mode;
- JSON repair chỉ sửa schema/cú pháp, không chọn lại article;
- cache có lock để tránh hai process cùng ghi một file;
- không chạy đồng thời hai command trên cùng `--run-dir`.

## 13. Cấu Trúc Output

```text
outputs/runs/<run-name>/
|-- cache/          JSONL dùng để resume
|-- artifacts/      kết quả trung gian
|-- logs/           log riêng từng phase
|-- submissions/    results.json và submission.zip
|-- manifest.json   fingerprint và trạng thái phase
`-- errors.jsonl    lịch sử lỗi/retry của orchestrator
```

Submission cuối:

```text
outputs/runs/<run-name>/submissions/final_answers_rescue_depth4/results.json
outputs/runs/<run-name>/submissions/final_answers_rescue_depth4/submission.zip
```

Submission phase 02-10 có `answer` rỗng và dùng để đo article retrieval. Phase
11 chứa article IDs và câu trả lời cuối.

## 14. Checklist Trước Full Run

1. `.env` có đúng Qdrant credentials và ba GPU endpoint.
2. ADC còn hiệu lực.
3. Corpus có đúng `82.570` articles.
4. BM25 được build từ chính corpus đó.
5. Qdrant dùng Qwen3 embedding, dimension 4096 và đủ expected IDs.
6. `--preflight-only` thành công.
7. Chọn `--run-dir` riêng cho run mới.
8. Không có process khác đang ghi cùng cache.
9. Khi resume, giữ nguyên input/corpus/BM25/config.
10. Chỉ nộp khi stage hoàn thành đủ question IDs và không có technical fallback.
