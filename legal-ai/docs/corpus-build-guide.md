# Hướng dẫn thu thập & xây dựng Corpus (TIP-CORPUS-001)

Tài liệu này ghi lại **toàn bộ quy trình** thu thập và xử lý dữ liệu để mở rộng corpus
từ **1.044 → 113.508 Điều luật**, phục vụ tăng recall cho bài toán IR (F2 macro).

> TL;DR — chạy lại từ đầu:
> ```bash
> python corpus/hf_download.py          # tải 3 file nhỏ + tier-A content
> python corpus/hf_fetch_large.py       # tải riêng file 3.5GB (tier-B content)
> python corpus/hf_ingest.py            # build corpus.json (0 ∪ A ∪ B)
> python corpus/coverage_probe.py --compare \
>     --old corpus/data/corpus_v1_1044.json \
>     --new corpus/data/corpus.json \
>     --questions ../R2AIStage1DATA.json   # đo độ phủ trước/sau
> ```

---

## 1. Vì sao phải mở rộng corpus?

Corpus gốc chỉ có **1.044 Điều từ 13 văn bản** (crawl tay). Phân tích bộ test 2.000 câu
cho thấy gold tham chiếu tới **~30+ luật** mà corpus cũ không có (Sở hữu trí tuệ, Luật
Thương mại, Kế toán, Bảo hiểm xã hội, An toàn vệ sinh lao động, Bảo vệ quyền lợi người
tiêu dùng…). Đây là **nút thắt recall** → cần một corpus phủ rộng các lĩnh vực SME.

Kết quả: **24/25** luật còn thiếu đã được bổ sung (chỉ `45/2020/QH14` không có trong nguồn).

---

## 2. Nguồn dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| Dataset | `th1nhng0/vietnamese-legal-documents` (HuggingFace) |
| License | **CC BY 4.0** (cho phép dùng — "open dataset") |
| Nguồn gốc | vbpl.vn — Cổng thông tin điện tử Bộ Tư pháp |
| DOI | 10.57967/hf/8598 |
| Định dạng | Parquet |

Dataset có 4 file parquet chính, chia **2 tầng**:

| File | Kích thước | Số dòng | Trường chính |
|------|-----------|---------|--------------|
| `data/metadata.parquet` (tier A meta) | 14 MB | 153.420 | `so_ky_hieu`, `loai_van_ban`, `linh_vuc`, `nganh`, `tinh_trang_hieu_luc` |
| `data/content.parquet` (tier A content) | 412 MB | 178.665 | `id`, `content_html` (HTML) |
| `legacy/metadata.parquet` (tier B meta) | 51 MB | 518.601 | `document_number`, `legal_type`, `legal_sectors`, `effect_status` |
| `legacy/content.parquet` (tier B content) | **3.5 GB** | 518.235 | `id`, `content` (plain text) |

- **Tier A** = lõi (metadata tiếng Việt giàu, HTML sạch, có hiệu lực).
- **Tier B** = recall net (metadata tiếng Anh, plain text, phủ rộng hơn).

---

## 3. Chuẩn bị môi trường

Trong virtual environment (xem [`usage.md`](usage.md)):

```bash
pip install datasets pyarrow beautifulsoup4 huggingface_hub requests
```

> `pyarrow`, `beautifulsoup4`, `requests` là các dependency mới cho bước build corpus
> (không cần cho pipeline runtime).

---

## 4. Quy trình 4 bước

```
hf_download.py + hf_fetch_large.py   →  hf_enumerate.py  →  hf_ingest.py  →  coverage_probe.py
        (tải parquet, cache)             (soi giá trị lọc)     (build corpus)     (đo độ phủ)
```

### Bước 1 — Tải dữ liệu (resumable, có cache)

```bash
# 3 file nhỏ + tier-A content (412MB). Đặt timeout để stall tự retry.
HF_HUB_DOWNLOAD_TIMEOUT=30 python corpus/hf_download.py

# File 3.5GB (tier-B content) tải riêng bằng downloader chia khối, chịu được rớt mạng
python corpus/hf_fetch_large.py
```

- File được cache trong `corpus/data/hf_cache/` (đã **gitignore**, ~4GB — không commit).
- `hf_download.py` dùng `hf_hub_download` (tự resume từ phần đã tải).
- `hf_fetch_large.py` dùng **HTTP Range theo khối 16MB**: mỗi khối có timeout + retry
  riêng, nên khi mạng bị ngắt chỉ mất khối hiện tại, không mất cả file. Dùng cho file
  3.5GB vì đường truyền hay bị cắt kết nối ở ~250MB.

> **Lưu ý kỹ thuật:** KHÔNG dùng `datasets.load_dataset(...streaming=True)` để đọc nội
> dung — nó ném `ArrowInvalid: large_string -> string` trên cột content quá lớn. Ta đọc
> parquet **trực tiếp bằng pyarrow** (xem bước 3).

### Bước 2 — Soi giá trị thực để đặt bộ lọc (tùy chọn)

```bash
python corpus/hf_enumerate.py
```

In ra phân bố thực tế của `loai_van_ban`, `tinh_trang_hieu_luc`, `linh_vuc` (tier A) và
`legal_type`, `effect_status`, `legal_sectors` (tier B). Nhờ bước này, các danh sách lọc
trong `hf_config.json` được đặt **dựa trên dữ liệu thật**, không phỏng đoán. Ví dụ phát hiện:
- `tinh_trang_hieu_luc` đa phần là "Hết hiệu lực toàn bộ" (80K) → phải loại.
- `loai_van_ban` có biến thể hoa/thường ("Nghị quyết" vs "Nghị Quyết") → cần chuẩn hóa.
- `linh_vuc` rỗng (None) ở ~70% → không lọc cứng theo lĩnh vực, mà match keyword trên `title`.

### Bước 3 — Build corpus

```bash
python corpus/hf_ingest.py                 # build đủ tier 0 ∪ A ∪ B
python corpus/hf_ingest.py --tier A        # chỉ tier A (debug)
python corpus/hf_ingest.py --limit-docs 50 --no-write   # smoke test nhanh
```

Ghi ra:
- `corpus/data/corpus.json` — corpus mở rộng (v2.0). **Gitignore** (≈780MB, để local).
- `corpus/data/corpus_v1_1044.json` — backup corpus gốc (tự tạo lần đầu, được commit).
- `corpus/data/SOURCES.md` — provenance: nguồn, license, số lượng, mapping loại VB, bộ lọc.

### Bước 4 — Đo độ phủ (proxy-coverage)

```bash
python corpus/coverage_probe.py --compare \
    --old corpus/data/corpus_v1_1044.json \
    --new corpus/data/corpus.json \
    --questions ../R2AIStage1DATA.json
```

Build BM25 (tái dùng `retrieval/bm25_index.py`) trên cả hai corpus, với mỗi câu trong 2.000
câu test lấy điểm BM25 **top-1**. So sánh bằng **ngưỡng tuyệt đối dùng chung** (p10 của
corpus cũ) → gap-rate mới thấp hơn nghĩa là dễ truy hồi hơn.

> ⚠️ **Caveat:** đây là proxy "retrievability", KHÔNG phải recall thật (không có gold).
> Chỉ dùng để so sánh tương đối cũ vs mới. Probe trên 113K Điều chạy khá lâu (~30 phút,
> BM25 thuần Python) — chạy nền, không đặt timeout ngắn.

---

## 5. Kiến trúc xử lý

### 5.1. Hợp nhất 3 tầng (union), ưu tiên 0 > A > B

```
Tier 0 (gốc 1044, ƯU TIÊN)  ─┐
Tier A (HF data/, HTML)      ─┼─►  dedup theo (law_id | Điều X)  ─►  corpus.json
Tier B (HF legacy/, text)    ─┘        (first-seen thắng)
```

- **Tier 0** = corpus gốc `corpus_v1_1044.json`, nạp **đầu tiên** → giữ nguyên 13 luật lõi
  (Quản lý thuế, Bộ luật Lao động…) với citation khớp gold. *Đây là lý do phải union thay
  vì thay thế: nếu chỉ lấy HF tier-A sẽ mất 6 luật lõi.*
- **Tier A** rồi **Tier B** chỉ **thêm** các Điều chưa có (key trùng → bỏ).
- Khóa dedup = `law_id | dieu_number` (chuẩn hóa thường). Lần build cuối loại 45.712 bản trùng.

### 5.2. Bộ lọc (config-driven trong `hf_config.json`)

| Bộ lọc | Tier A | Tier B |
|--------|--------|--------|
| **Hiệu lực** | giữ `Còn hiệu lực`, `Hết hiệu lực một phần` | giữ `In effect` |
| **Loại VB** | giữ Bộ luật/Luật/Pháp lệnh/Hiến pháp/Nghị định/Nghị quyết/Thông tư/VB hợp nhất (+ liên tịch) | map Anh→Việt (Law→Luật, Decree→Nghị định…) rồi áp cùng danh sách |
| **Cấp TW** | loại mã chứa `HĐND`/`UBND`/`HU`/`TU` (văn bản địa phương) | như tier A |
| **Lĩnh vực SME** | match ~82 keyword trên `title + linh_vuc + nganh` | match keyword trên `title + legal_sectors` |

Loại VB cá biệt (Quyết định, Chỉ thị, Công văn…) và VB địa phương đều bị loại. Toàn bộ
mapping "giữ/bỏ" được ghi vào `SOURCES.md` để review.

### 5.3. Parse HTML/text → Điều

- Tier A `content_html` → BeautifulSoup `get_text()` → text.
- Tier B `content` → dùng trực tiếp (đã là plain text).
- Cả hai đưa vào **`LegalTextParser` cũ (không sửa)** — giữ nguyên fix regex "quy" và
  schema Article. Mỗi Điều = 1 Article (`law_id, law_type, law_name, dieu_number,
  dieu_title, content, khoan_list, chunk_id, relevant_doc_str, relevant_article_str`).
- File content rỗng (chỉ scan PDF) → bỏ qua, đếm vào thống kê.

### 5.4. Định dạng citation (ăn/thua điểm IR)

Hàm format có **flag** `citation.name_variant` trong config:
- `loai_title` (mặc định): `<mã>|<Loại> <Trích yếu>` — khớp ví dụ gold/Dashboard
  (vd `04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa`).
- `loai_makh_title`: `<mã>|<Loại> <mã> <Trích yếu>` — theo công thức chữ trong thể lệ.

> **Đặc tả BTC mâu thuẫn:** phần chữ ghi "Loại + **Mã** + Trích yếu" nhưng **ví dụ** lại
> "Loại + Trích yếu" (không có mã). Ngoài ra `eval_set` (gold) và ví dụ Dashboard còn
> lệch hoa/thường cùng một văn bản ("quy định" vs "Quy định") → suy ra hệ thống **chấm
> chuẩn hóa theo `mã văn bản | điều`, KHÔNG so chuỗi tên nguyên văn**. Vì vậy mặc định
> theo **ví dụ** (`loai_title`) là an toàn nhất.

**Chuẩn hóa tên** (`citation.normalize_names_from_hf: true`, mặc định bật): với mỗi Điều,
ghi đè `law_name` bằng **trích yếu chính thức đầy đủ** lấy từ metadata HF (ưu tiên tier A),
bỏ tiền tố loại + viết hoa chữ cái đầu; **giữ nguyên `law_type`** (đáng tin hơn theo từng
tier). Bước này khắc phục:
- Tên viết tắt của corpus gốc: `Nghị định Hỗ trợ DNNVV` → `Nghị định Quy định chi tiết và
  hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa`.
- Tên thường-hoa lệch của HF: `Luật hỗ trợ…` → `Luật Hỗ trợ…`.

> Bước này **không đổi điểm IR** (BM25 chỉ index `dieu_title + content`, không index tên;
> hệ thống chấm theo `mã + điều`) — chỉ làm `relevant_docs`/`relevant_articles` đúng định
> dạng trích yếu như ví dụ BTC.

---

## 6. File cấu hình `hf_config.json`

Mọi tham số đều nằm trong config (không hardcode trong code):

| Khóa | Ý nghĩa |
|------|---------|
| `cache_dir`, `tiers.*` | đường dẫn parquet, bật/tắt từng tier |
| `filters.effect_keep` | trạng thái hiệu lực giữ lại (A/B) |
| `filters.doc_type_keep_vi` | danh sách loại VB giữ (tiếng Việt) |
| `filters.legal_type_map_en_vi` | map loại VB Anh→Việt (tier B) |
| `filters.exclude_code_patterns` | mẫu mã loại VB địa phương (HĐND/UBND…) |
| `filters.domain_keywords` | ~82 keyword lĩnh vực SME |
| `filters.domain_apply` | bật lọc lĩnh vực cho A/B |
| `citation.name_variant` | biến thể định dạng tên citation (`loai_title` / `loai_makh_title`) |
| `citation.normalize_names_from_hf` | ghi đè `law_name` bằng trích yếu chính thức HF (giữ `law_type`) |
| `dedup.prefer_tier` | thứ tự ưu tiên dedup |
| `output.drop_khoan_list` | bỏ `khoan_list` (trùng content, không dùng downstream) → corpus nhẹ ~½ |

---

## 7. Kết quả

| Chỉ số | Trước | Sau |
|--------|-------|-----|
| Số Điều | 1.044 | **113.508** |
| Số văn bản | 13 | **6.849** |
| Luật gold còn thiếu được phủ | — | **24/25** |
| gap-rate (ngưỡng chung) | 0.100 | **0.011** |
| median top-1 BM25 | 60.0 | **76.1** |
| mean top-1 BM25 | 62.1 | **78.2** |

Build cuối: tier 0 = 1.044, tier A = 56.441, tier B = 56.023, dedup loại 45.712.

---

## 8. Lưu ý vận hành & các quyết định còn treo

- **corpus.json ≈780MB** → vượt giới hạn 100MB của GitHub, đang **gitignore + để local**.
  Đặt `output.drop_khoan_list: true` (mặc định) → build lại còn ~390MB (`khoan_list` là dữ
  liệu thừa, không component nào đọc).
- **Qdrant Cloud đang lệch:** collection `legal_vn_vertex` vẫn là 1.044 điểm cũ. Muốn dùng
  corpus mới trên backend Vertex phải re-embed:
  ```bash
  python setup_qdrant_cloud.py --config config_vertex.yaml \
      --corpus corpus/data/corpus.json --force
  ```
  (embed 113K Điều — tốn chi phí/thời gian; cân nhắc thu hẹp tier B trước). BM25 index tự
  rebuild từ corpus.json nên không cần thao tác thủ công.
- Tài liệu cũ (`usage.md`, README) vẫn ghi "1044 articles / 13 văn bản" — cập nhật khi
  chốt chiến lược corpus.

---

## 9. Sự cố thường gặp

| Triệu chứng | Nguyên nhân / xử lý |
|-------------|---------------------|
| `ArrowInvalid: large_string -> string` | Đừng dùng `datasets` streaming; đọc parquet bằng pyarrow (đã làm trong `hf_ingest.py`). |
| Tải 3.5GB đứng ở ~250MB | Đường truyền cắt kết nối dài. Dùng `hf_fetch_large.py` (Range theo khối). Có thể seed `.part` từ file `.incomplete` của lần tải trước để không tải lại. |
| `hf_download.py` treo khi stall | Đặt `HF_HUB_DOWNLOAD_TIMEOUT=30` để socket timeout kích retry. |
| Probe chạy rất lâu / bị timeout | BM25 thuần Python trên 113K Điều × 2000 câu ~30 phút. Chạy nền, đừng đặt timeout ngắn. |
| Thiếu file parquet khi build | Chạy lại bước 1 (download có resume, không tải lại phần đã có). |
