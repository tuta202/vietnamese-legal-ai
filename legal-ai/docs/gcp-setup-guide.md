# GCP Setup Guide — Vietnamese Legal AI v1

## Tổng quan hạ tầng

```
┌─────────────────────────────────────────────────────┐
│  GCP VM: g2-standard-8 (1x L4 24GB, 8 vCPU, 32GB)  │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Qdrant      │  │ vLLM Server  │  │ Pipeline   │  │
│  │ Docker      │  │ port 8000    │  │ Python     │  │
│  │ port 6333   │  │              │  │            │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
│                                                     │
│  SSD: 200GB (models + data)                         │
└─────────────────────────────────────────────────────┘
```

## Tại sao chọn L4 24GB?

| GPU | VRAM | Giá on-demand | Giá spot | Ghi chú |
|-----|------|---------------|----------|---------|
| T4 | 16GB | ~$0.35/hr | ~$0.08/hr | Quá nhỏ cho Embedding 8B fp16 |
| **L4** | **24GB** | **~$0.70/hr** | **~$0.15/hr** | **Vừa đủ, tối ưu chi phí** |
| A100 40GB | 40GB | ~$3.67/hr | ~$1.10/hr | Thoải mái nhưng tốn gấp 5x |

L4 24GB đủ cho:
- Phase 1 (embed): Qwen3-Embedding-8B fp16 ≈ 16GB → vừa L4
- Phase 2 (inference): Qwen2.5-7B-Instruct AWQ 4-bit ≈ 5GB + Qwen3-Reranker-4B fp16 ≈ 8GB = 13GB → thoải mái

**Ước tính chi phí toàn cuộc thi:**

| Phase | Thời gian | Cost (on-demand) | Cost (spot) |
|-------|-----------|-------------------|-------------|
| Setup + embed corpus | 3 giờ | $2.10 | $0.45 |
| Dev + test (4h/ngày × 20 ngày) | 80 giờ | $56.00 | $12.00 |
| Final batch inference | 5 giờ | $3.50 | $0.75 |
| **Tổng** | **88 giờ** | **~$62** | **~$13** |

Dư $238–$287 buffer. Khuyến nghị: dùng **spot VM** cho dev, on-demand cho final submission.

---

## Bước 1: Tạo VM

### 1.1 Qua Console (UI)

1. Vào [Google Cloud Console](https://console.cloud.google.com)
2. **Compute Engine → VM Instances → Create Instance**
3. Cấu hình:

```
Name:           legal-ai-v1
Region:         us-central1 (Iowa) — giá rẻ nhất
Zone:           us-central1-a (hoặc -b, -c — check GPU availability)
Machine type:   g2-standard-8
                  → 1x NVIDIA L4 (24GB)
                  → 8 vCPU
                  → 32 GB RAM

Boot disk:
  OS:           Ubuntu 22.04 LTS
  Type:         SSD persistent disk
  Size:         200 GB
  (3 models ≈ 60GB download, Qdrant data ≈ 5GB, còn lại cho OS + pip)

Provisioning:   Spot (tiết kiệm 80%)
                ⚠️ Spot có thể bị thu hồi — checkpoint thường xuyên
                Dùng On-demand cho final run nếu cần stability

Firewall:       Allow HTTP traffic ✓ (nếu cần remote access)
```

### 1.2 Qua gcloud CLI

```bash
# Tạo VM spot (rẻ nhất)
gcloud compute instances create legal-ai-v1 \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=count=1,type=nvidia-l4 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE

# SSH vào VM
gcloud compute ssh legal-ai-v1 --zone=us-central1-a
```

> **Lưu ý:** Nếu zone `us-central1-a` hết GPU, thử `-b`, `-c`, hoặc đổi sang `us-east1-c`, `us-west1-b`.

---

## Bước 2: Setup môi trường trên VM

SSH vào VM rồi chạy lần lượt:

### 2.1 NVIDIA Driver + CUDA

```bash
# Kiểm tra GPU đã nhận chưa
nvidia-smi
# Nếu chưa có nvidia-smi, cài driver:
sudo apt-get update
sudo apt-get install -y nvidia-driver-550
sudo reboot
# Sau reboot, SSH lại, verify:
nvidia-smi
# Expect: NVIDIA L4, 24GB VRAM
```

### 2.2 Docker (cho Qdrant)

```bash
# Cài Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

### 2.3 Python + Dependencies

```bash
# Python 3.11 (Ubuntu 22.04 mặc định có 3.10, cần upgrade)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Tạo venv
python3.11 -m venv ~/legal-ai-env
source ~/legal-ai-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece
pip install vllm
pip install qdrant-client
pip install rank-bm25
pip install pyyaml
pip install openai   # cho vLLM OpenAI-compatible API

# Verify CUDA trong Python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expect: CUDA: True, GPU: NVIDIA L4
```

### 2.4 Clone project

```bash
cd ~
# Upload project từ local hoặc clone từ git
# Option A: gcloud SCP
# (chạy trên máy local):
#   gcloud compute scp --recurse ./legal-ai legal-ai-v1:~/legal-ai --zone=us-central1-a

# Option B: git clone (nếu đã push)
#   git clone https://github.com/YOUR_REPO/legal-ai.git

cd ~/legal-ai
```

---

## Bước 3: Start Qdrant

```bash
# Pull và start Qdrant (chạy background)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Verify
curl http://localhost:6333/healthz
# Expect: {"title":"qdrant - vectorass engine","version":"..."}

# Qdrant data persist ở ~/qdrant_storage
# Nếu VM bị stop/restart, chạy lại docker start qdrant
```

---

## Bước 4: Download models (một lần, ~30 phút)

```bash
source ~/legal-ai-env/bin/activate

# Download 3 models vào cache (không load vào GPU)
python -c "
from huggingface_hub import snapshot_download

# 1. Embedding model (~16GB)
print('Downloading Qwen3-Embedding-8B...')
snapshot_download('Qwen/Qwen3-Embedding-8B')

# 2. Reranker (~8GB)
print('Downloading Qwen3-Reranker-4B...')
snapshot_download('Qwen/Qwen3-Reranker-4B')

# 3. LLM — dùng AWQ 4-bit để tiết kiệm VRAM
# AWQ cho phép Reranker + LLM cùng chạy trên L4
print('Downloading Qwen2.5-7B-Instruct-AWQ...')
snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ')
"

# Models cache tại ~/.cache/huggingface/hub/
# Verify
du -sh ~/.cache/huggingface/hub/models--Qwen*
```

> **Tại sao AWQ cho LLM?**
> L4 chỉ có 24GB. Trong phase inference, cần chạy đồng thời:
> - Qwen3-Reranker-4B fp16 ≈ 8GB
> - Qwen2.5-7B-Instruct ≈ 14GB fp16 → KHÔNG VỪA (tổng 22GB + overhead)
> 
> Dùng AWQ 4-bit: Qwen2.5-7B ≈ 5GB → tổng 13GB → thoải mái.
> Chất lượng AWQ gần bằng fp16 (mất ~1-2% trên benchmark).

---

## Bước 5: Embed corpus vào Qdrant (một lần)

Bước này load Qwen3-Embedding-8B fp16 (16GB) **một mình** lên GPU,
embed toàn bộ 1044 articles, upsert vào Qdrant. Sau đó unload model
để trả VRAM cho inference.

```bash
source ~/legal-ai-env/bin/activate
cd ~/legal-ai

# Embed corpus — chạy ~10-15 phút
python retrieval/embedder.py

# Verify Qdrant collection
curl http://localhost:6333/collections/legal_vn
# Expect: "points_count": 1044

# Sau khi embed xong, model tự unload khỏi GPU
# Verify VRAM trống:
nvidia-smi
# Expect: ~0 MB GPU memory used
```

---

## Bước 6: Start vLLM server

```bash
source ~/legal-ai-env/bin/activate

# Start vLLM phục vụ Qwen2.5-7B-Instruct-AWQ
# AWQ 4-bit ≈ 5GB VRAM — còn đủ cho Reranker
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.35 \
  --dtype auto \
  --quantization awq &

# Đợi model load xong (~2 phút), verify:
curl http://localhost:8000/v1/models
# Expect: {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct-AWQ", ...}]}

# Test một query:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Xin chào"}],
    "max_tokens": 50
  }'
```

> **Lưu ý `--gpu-memory-utilization 0.35`:**
> Giới hạn vLLM chỉ dùng 35% VRAM (≈8.4GB). Để lại ~15GB cho Reranker
> khi pipeline chạy. Nếu Reranker OOM, giảm xuống 0.30.

---

## Bước 7: Update config

Sửa `retrieval/config.yaml` cho đúng endpoint:

```yaml
qdrant:
  host: localhost
  port: 6333
  collection: legal_vn

vllm:
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-7B-Instruct-AWQ   # ← AWQ version

models:
  embedder: Qwen/Qwen3-Embedding-8B
  reranker: Qwen/Qwen3-Reranker-4B

retrieval:
  top_k_dense: 20
  top_k_bm25: 20
  top_k_fusion: 20
  top_k_rerank: 7
  rrf_k: 60

generator:
  temperature: 0.3
  max_tokens: 2048
  top_p: 0.9
```

---

## Bước 8: Smoke test end-to-end

```bash
cd ~/legal-ai
source ~/legal-ai-env/bin/activate

# Test 3 câu hỏi mẫu (KHÔNG dùng mock)
python pipeline.py --input tests/sample_questions.json

# Validate output
python submit.py --validate-only results.json
# Expect: "VALID: 3 entries, 0 errors"

# Tạo submission
python submit.py --input results.json --output submission/
# → submission/submission.zip ready to upload
```

---

## Bước 9: Chạy với test set cuộc thi

Khi Ban tổ chức phát test set:

```bash
# 1. Download test_questions.json từ dashboard
# 2. Copy vào VM
gcloud compute scp test_questions.json legal-ai-v1:~/legal-ai/ \
  --zone=us-central1-a

# 3. SSH vào VM, chạy pipeline
cd ~/legal-ai
python pipeline.py --input test_questions.json

# 4. Validate + build submission
python submit.py --input results.json --output submission/
# → submission.zip

# 5. Copy submission về local
gcloud compute scp legal-ai-v1:~/legal-ai/submission/submission.zip . \
  --zone=us-central1-a

# 6. Nộp lên http://leaderboard.aiguru.com.vn/
```

---

## Quản lý chi phí

### Tắt VM khi không dùng

```bash
# Từ local — QUAN TRỌNG: L4 tính tiền khi VM đang chạy dù idle
gcloud compute instances stop legal-ai-v1 --zone=us-central1-a

# Bật lại khi cần
gcloud compute instances start legal-ai-v1 --zone=us-central1-a

# Sau khi bật lại, cần restart services:
# SSH vào VM rồi:
docker start qdrant
source ~/legal-ai-env/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.35 --quantization awq &
```

### Monitor chi phí

```bash
# Check billing estimate
gcloud billing accounts list
# Hoặc vào Console → Billing → Reports
```

### Xóa VM khi xong cuộc thi

```bash
gcloud compute instances delete legal-ai-v1 --zone=us-central1-a
docker volume rm qdrant_storage  # nếu cần
```

---

## Troubleshooting

### GPU không có sẵn ở zone đã chọn

```
ERROR: ZONE_RESOURCE_POOL_EXHAUSTED
```
→ Đổi zone: `us-central1-b`, `us-central1-c`, `us-east1-c`, `us-west1-b`

### Spot VM bị thu hồi

```
WARNING: Instance was preempted
```
→ Data vẫn ở disk (persistent). Start lại VM, restart services.
→ Nếu bị thu hồi thường xuyên → chuyển on-demand cho final run.

### Reranker OOM khi pipeline chạy

```
torch.cuda.OutOfMemoryError
```
→ Giảm vLLM: `--gpu-memory-utilization 0.25`
→ Hoặc: load reranker fp16 trước, chạy rerank, rồi unload trước khi gọi vLLM
→ Hoặc: dùng Qwen3-Reranker-0.6B (nhỏ hơn, vẫn tốt)

### vLLM không start được

```
ValueError: AWQ quantization is not supported
```
→ Verify vllm version ≥ 0.4.0: `pip install vllm --upgrade`
→ Nếu AWQ lỗi → dùng GPTQ: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`

### Embedding quá chậm

1044 articles nên embed trong 10-15 phút trên L4.
Nếu lâu hơn → check batch_size (tăng lên 32 nếu VRAM đủ).

---

## VRAM Budget trên L4 24GB

```
Phase 1 — Embedding (one-time, sequential):
  ┌──────────────────────────────────┬──────────┐
  │ Qwen3-Embedding-8B fp16         │  ~16 GB  │
  │ Overhead (PyTorch, CUDA)        │   ~2 GB  │
  │ Còn lại                         │   ~6 GB  │
  └──────────────────────────────────┴──────────┘
  → Vừa đủ. Chạy embedding xong → unload.

Phase 2 — Inference (concurrent):
  ┌──────────────────────────────────┬──────────┐
  │ vLLM Qwen2.5-7B-AWQ 4-bit      │  ~5 GB   │
  │ vLLM KV cache (max 4096 tokens) │  ~3 GB   │
  │ Qwen3-Reranker-4B fp16         │  ~8 GB   │
  │ Overhead                        │  ~2 GB   │
  │ Còn lại                         │  ~6 GB   │
  └──────────────────────────────────┴──────────┘
  → Thoải mái. Cả hai model chạy đồng thời.
```