# RAG Workflow Hiện Tại

Workflow hiện tại dùng hai tầng LLM verifier, tối ưu riêng cho Gemma 3 12B-it.
Stage 2 cũ đã được loại bỏ hoàn toàn.

```mermaid
flowchart TD
    Q["Câu hỏi"] --> RW["LLM call 1: Query rewrite"]
    Q --> DI["LLM call 2: Legal intent decomposition"]

    RW --> GR["Global retrieval: BM25 + Dense query + Dense topic"]
    GR --> RRF["RRF global top 60"]

    DI --> IR["Per-intent retrieval: BM25 + Dense"]
    IR --> I10["RRF top 10 mỗi intent"]

    RRF --> KEEP["Keep = top60 RRF ∪ intent hits"]
    I10 --> KEEP
    KEEP --> BGE["BGE rerank theo từng intent"]

    RRF --> TIER["Tiered union"]
    BGE --> TIER
    I10 --> TIER
    TIER --> S1["Stage 1 compact: recall-oriented"]
    S1 --> CLEAN["Penalty-aware conservative cleanup"]
    CLEAN --> FINAL["Final collective compact: precision-oriented"]
    FINAL --> GATE["Enforcement-role gate"]
    GATE --> GEN["Grounded answer generation"]
    GEN --> OUT["results.json + submission.zip"]
```

## 1. Query Analysis

Query analysis gồm hai LLM call độc lập:

1. Query rewrite sinh `rewritten_query` và `topic_description`.
2. Legal intent decomposition sinh các legal intent độc lập.

Các call lỗi hoặc parse lỗi không được ghi fallback vào cache. Pipeline để lại câu
lỗi và chạy lại bằng resume.

## 2. Global Và Intent Retrieval

Global retrieval:

```text
BM25(rewritten_query), top 350
Dense(rewritten_query), top 350
Dense(topic_description), top 350
-> RRF
-> top60_rrf
```

Intent retrieval cho từng intent:

```text
BM25(intent), top 50
Dense(intent), top 50
-> RRF
-> top10_each_intent
```

Keep reservoir:

```text
keep = top60_rrf | intent_hits
```

## 3. BGE Và Tiered Union

BGE rerank toàn bộ keep theo từng legal intent. Candidate đưa sang verifier là
union không cap cứng:

```text
top12_rrf_global
| top5_bge_each_intent
| top5_raw_intent_each_intent
```

Artifact phase 05:

```text
submissions/tiered_rrf12_bge5_rawintent5/submission.zip
```

## 4. Stage 1 Compact Verifier

Vai trò: nén mạnh candidate pool nhưng vẫn ưu tiên recall.

```text
model              = Gemma 3 12B-it
candidate format   = compact alias A1, A2, ...
content            = tối đa 1.800 ký tự/article
batch size         = 6 articles/call
temperature        = 0
strict errors      = true
prompt version     = gemma-recall-v4
```

Mỗi batch được đánh giá độc lập. Prompt không ép một batch phải cover tất cả
intent và cho phép trả danh sách rỗng nếu batch không có evidence. Article chỉ
cùng chủ đề nhưng không chứa quy định cụ thể phải bị loại. Nếu excerpt bị thiếu
nhưng có khả năng chứa evidence cần thiết thì giữ để bảo vệ recall.

Không có vòng global ở Stage 1 vì vòng này dễ làm mất evidence vừa được giữ từ
các batch khác. Output giữ thứ tự evidence ban đầu.

Nếu model trả JSON sai schema, hệ thống gọi một JSON-repair giới hạn chỉ để chuẩn
hóa selection đã có. Repair không được chọn lại article. Nếu vẫn lỗi, question
không được cache và phải chạy resume.

Artifact mới:

```text
cache/stage1_gemma_compact_v4.jsonl
submissions/stage1_gemma_compact_v4/submission.zip
```

Prompt này thay thế prompt cũ khiến Gemma giữ khoảng 71,6% candidate. Kết quả mới
chưa được benchmark vì workflow chưa được chạy lại.

## 5. Penalty-Aware Cleanup

Sau Stage 1, deterministic cleanup chỉ loại article xử phạt/cưỡng chế rõ ràng khi
question không hỏi chế tài và đã có evidence nghiệp vụ thay thế. Rule không drop
rộng chỉ dựa trên keyword.

Hiệu lực văn bản đã được xử lý trong corpus as-of nên phase này chạy với
`--skip-invalid`.

## 6. Final Collective Compact Verifier

Vai trò: so sánh toàn bộ shortlist và ưu tiên precision.

```text
model              = Gemma 3 12B-it
candidate format   = compact alias A1, A2, ...
content            = tối đa 2.200 ký tự/article
batch size         = 6
direct max         = 8
minimum input size = 2
preserve top1      = false
temperature        = 0
strict errors      = true
prompt version     = gemma-precision-v5
```

Chiến lược call:

- Dưới 2 article: giữ nguyên, không gọi LLM.
- Từ 2 đến 8 article: một collective call để so sánh trực tiếp.
- Trên 8 article: chia batch 6, union shortlist của các batch, sau đó gọi một
  global collective round để khử nhiễu và trùng lặp giữa batch.

Final chỉ giữ tập nhỏ nhất nhưng đủ cover mọi legal intent. Article chỉ cùng chủ
đề, sai subtask, meta chung, lặp cùng vai trò hoặc điều xử phạt không được hỏi sẽ
bị loại. Các article bổ sung quy tắc pháp lý khác nhau vẫn được giữ.

Một selection rỗng hợp lệ được bảo vệ bằng top1 evidence. Lỗi request/parse trong
strict mode không fallback và không cache. Final cũng dùng JSON repair giống
Stage 1.

## 7. Enforcement-Role Gate

Gate hậu xử lý chỉ loại điều xử phạt/cưỡng chế khi vai trò đó rõ ràng không phù
hợp câu hỏi. Đây là rule hẹp, không phải bộ lọc relevance tổng quát.

## 8. Answer Generation

Generation nhận toàn bộ article sau gate. Mỗi article dùng tối đa 2.400 ký tự;
article dài lấy excerpt theo formatter hiện tại và tổng content budget là 48.000
ký tự. Prompt yêu cầu câu trả lời tiếng Việt, grounded vào article đã chọn và
không chuyển điều kiện/hệ quả giữa các khoản, điểm lân cận.

## Best hiện tại: raw-intent top1 coverage rescue

Sau `09_enforcement_role_gate`, bản thử nghiệm tốt nhất hiện tại áp dụng một post-process
deterministic để khôi phục coverage bị Final verifier loại quá tay. Bản này chưa được tích hợp
thành phase mặc định của pipeline.

Với từng legal intent:

1. Xét article raw-intent rank 1.
2. Bỏ qua nếu final còn ít nhất một article thuộc raw-intent top 3 của intent đó.
3. Chỉ rescue nếu article rank 1 đã sống sót qua Stage 1 và penalty cleanup.
4. Giữ nguyên thứ tự final, append article rescue theo thứ tự intent và không thêm trùng.

Kết quả trên leaderboard:

| Version | Precision | Recall | F2 xấp xỉ | Mean article |
|---|---:|---:|---:|---:|
| Enforcement-role gate | 0.4638 | 0.7210 | 0.6490 | 3.5340 |
| Raw-intent top1 coverage rescue | **0.4788** | **0.7510** | **0.6743** | 3.6855 |

Post-process thay đổi 275/2.000 question và rescue 303 article. Artifact:

```text
submissions/best_final_enforcement_gate_rawintent_top1_rescue/submission.zip
submissions/best_final_enforcement_gate_rawintent_top1_rescue/diagnostics.json
```

Đây là bản best hiện tại của nhánh GPU. Audit thủ công vẫn phát hiện một số raw top1 nhiễu,
vì vậy bước cải tiến tiếp theo là thử guard `raw-intent top1 ∩ BGE top3/top5 cùng intent` và
duplicate cũ-mới chỉ khi có quan hệ thay thế đã được xác minh.

## 9. Phase Hiện Tại

| Phase | Nội dung |
|---|---|
| `01_query_analysis` | Rewrite + intent decomposition |
| `02_global_rrf` | Global retrieval/RRF |
| `03_raw_intent_retrieval` | Per-intent BM25 + Dense + RRF |
| `04_bge_intent_rerank` | BGE theo intent |
| `05_tiered_union` | Recall-oriented union |
| `06_stage1_compact` | Gemma recall-oriented compact verifier |
| `07_penalty_cleanup` | Deterministic penalty cleanup |
| `08_final_collective` | Gemma precision-oriented compact verifier |
| `09_enforcement_role_gate` | Enforcement-role cleanup |
| `10_answer_generation` | Grounded final answer |

## 10. Resume Và Cache Safety

Mỗi phase phải đủ toàn bộ question mới được đánh dấu complete. Lỗi kỹ thuật hoặc
JSON không hợp lệ không được chuyển thành kết quả fallback. Chạy lại cùng command
và `--run-dir` để chỉ xử lý phần thiếu.

Khi chuyển một run cũ sang workflow hai verifier này, dùng một lần:

```text
--accept-code-change --accept-workflow-change
```

Nếu endpoint vừa redeploy, thêm `--accept-runtime-change`. Stage 1 dùng cache/path
v2 nên output cũ không bị tái sử dụng sau khi prompt thay đổi.
