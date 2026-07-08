# Reflex Legal QA UI

UI này dùng trực tiếp workflow production một-câu trong `legal_rag.production.single_question_runner.SingleQuestionRagRunner`.

## Mục đích

Giao diện không chỉ hiển thị câu trả lời dạng text. Mỗi lượt hỏi sẽ có:

- câu hỏi người dùng;
- kết luận ngắn gọn;
- phân tích pháp lý;
- danh sách văn bản nguồn;
- danh sách điều luật liên quan;
- cảnh báo/phạm vi áp dụng nếu grounding chưa đủ rõ.

## Chuẩn bị

Cài dependencies:

```powershell
cd C:\development\vietnamese-legal-ai\legal-ai
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Đảm bảo các endpoint/model trong `.env` hoặc môi trường đã được cấu hình giống pipeline production.

Các biến tùy chọn cho UI:

```powershell
$env:LEGAL_RAG_CONFIG="config_gpu_gemini_production.yaml"
$env:LEGAL_RAG_CORPUS="corpus\data\corpus_clean_asof_20260301.json"
$env:LEGAL_RAG_BM25_INDEX="retrieval\data\bm25_index_asof_20260301.pkl"
$env:LEGAL_RAG_EXPECTED_ARTICLES="82570"
$env:LEGAL_RAG_RESCUE_DEPTH="4"
```

Nếu không set các biến này, UI dùng các giá trị mặc định ở trên.

## Chạy UI

```powershell
cd C:\development\vietnamese-legal-ai\legal-ai
.\.venv\Scripts\reflex.exe run
```

Mặc định:

- frontend: `http://localhost:3000`
- backend Reflex: `http://localhost:8001`

## Luồng tích hợp

Khi người dùng bấm Tra cứu:

```text
Reflex UI
-> SingleQuestionRagRunner
-> rewrite + decompose song song
-> global RRF + raw intent retrieval song song
-> BGE intent-wise compression
-> Stage 1 verifier
-> penalty cleanup
-> final collective verifier
-> enforcement gate
-> raw-intent top1 coverage rescue
-> answer generation
-> UI hydrate article metadata/content từ corpus
```

UI giữ nguyên workflow production; chỉ thay đổi cách hiển thị kết quả.
