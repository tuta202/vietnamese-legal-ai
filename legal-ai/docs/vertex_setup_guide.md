# Hướng dẫn Setup GCP + Qdrant Cloud (thủ công)

> **Ai làm:** Con người (bạn)
> **Khi nào:** Trước khi giao TIP-VERTEX-002 cho Thợ
> **Thời gian:** ~15 phút

---

## Bước 1: Enable Vertex AI API trên GCP

1. Vào Google Cloud Console: https://console.cloud.google.com
2. Chọn project bạn muốn dùng (hoặc tạo mới)
3. Ghi lại **Project ID** (không phải Project Name)
4. Mở Cloud Shell hoặc terminal có gcloud CLI, chạy:

```bash
# Đặt project
gcloud config set project YOUR_PROJECT_ID

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Verify
gcloud services list --enabled | grep aiplatform
# Phải thấy: aiplatform.googleapis.com
```

**Lưu ý:** Free trial $300 credits đã bao gồm Vertex AI API calls. Không phát sinh phí thêm.

---

## Bước 2: Xác thực GCP trên máy local

Trên máy bạn sẽ chạy code (local hoặc VM):

```bash
# Cài gcloud CLI nếu chưa có:
# https://cloud.google.com/sdk/docs/install

# Login + set application default credentials
gcloud auth application-default login

# Sẽ mở browser → đăng nhập Google → authorize
# File credentials tự lưu tại:
#   Linux/Mac: ~/.config/gcloud/application_default_credentials.json
#   Windows:   %APPDATA%\gcloud\application_default_credentials.json
```

**Verify xác thực OK:**
```bash
gcloud auth application-default print-access-token
# Phải in ra 1 token dài, không lỗi
```

---

## Bước 3: Tạo Qdrant Cloud Free Cluster

1. Vào https://cloud.qdrant.io → Sign Up (miễn phí, không cần thẻ)
2. Sau khi đăng nhập → **Create Cluster**
3. Chọn:
   - **Cluster name:** `legal-vn` (hoặc tùy ý)
   - **Plan:** Free
   - **Cloud provider:** Google Cloud
   - **Region:** chọn gần nhất (ví dụ `us-central1` hoặc `europe-west4`)
4. Nhấn **Create** → đợi ~1 phút

### Lấy credentials:

**URL:**
- Vào cluster vừa tạo → tab **Overview** hoặc **Connect**
- Copy URL dạng: `https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east4-0.gcp.cloud.qdrant.io:6333`

**API Key:**
- Tab **API Keys** → **Create API Key**
- Copy key (chỉ hiển thị 1 lần, lưu lại ngay)

---

## Bước 4: Điền credentials vào config_vertex.yaml

Mở file `config_vertex.yaml` tại repo root, điền 3 chỗ:

```yaml
llm:
  gcp_project: "your-project-id"        # ← Project ID từ Bước 1
  gcp_location: "us-central1"           # ← giữ nguyên hoặc đổi theo region

qdrant:
  url: "https://xxx.cloud.qdrant.io:6333"  # ← URL từ Bước 3
  api_key: "your-qdrant-api-key"            # ← API Key từ Bước 3
```

---

## Bước 5: Kiểm tra nhanh (tùy chọn)

```bash
# Test GCP auth
python -c "
from google import genai
client = genai.Client(vertexai=True, project='YOUR_PROJECT_ID', location='us-central1')
r = client.models.generate_content(model='gemini-2.5-flash-lite', contents='Xin chào')
print('Vertex AI OK:', r.text[:50])
"

# Test Qdrant Cloud
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='YOUR_URL', api_key='YOUR_KEY')
print('Qdrant OK:', c.get_collections())
"
```

Nếu cả 2 test pass → giao TIP-VERTEX-002 cho Thợ.

---

## Chi phí ước tính

| Hạng mục | Chi phí |
|----------|---------|
| Vertex AI API (Gemini Flash-Lite) | ~$1-2 / lần chạy 50 questions |
| Gemini Embedding (1044 articles) | ~$0.05 / lần embed |
| Qdrant Cloud Free Tier | $0 |
| **Tổng cho toàn bộ thử nghiệm** | **< $10** (dư ~$290 credits) |

---

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|------|------------|-----|
| `403 Permission Denied` | Chưa enable API | `gcloud services enable aiplatform.googleapis.com` |
| `401 Unauthorized` | Chưa auth | `gcloud auth application-default login` |
| `Quota exceeded` | Rate limit Vertex AI | Đợi 1 phút, thử lại |
| `Could not connect to Qdrant` | URL sai hoặc thiếu port | Kiểm tra URL có `:6333` |
| `Invalid API key` | Key Qdrant sai | Tạo key mới trên Qdrant Cloud |