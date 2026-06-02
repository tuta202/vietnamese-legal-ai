# GCP Quick Reference — Legal AI v1

## Tạo VM (1 lần)
```bash
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
```

## Daily workflow
```bash
# Bật VM
gcloud compute instances start legal-ai-v1 --zone=us-central1-a

# SSH
gcloud compute ssh legal-ai-v1 --zone=us-central1-a

# Trong VM — restart services
docker start qdrant
source ~/legal-ai-env/bin/activate
cd ~/legal-ai
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.35 --quantization awq &

# Chạy pipeline
python pipeline.py --input test_questions.json
python submit.py --input results.json --output submission/

# TẮT VM KHI XONG (quan trọng — tiết kiệm tiền)
# Từ local:
gcloud compute instances stop legal-ai-v1 --zone=us-central1-a
```

## Copy file giữa local ↔ VM
```bash
# Local → VM
gcloud compute scp ./file.json legal-ai-v1:~/legal-ai/ --zone=us-central1-a
gcloud compute scp --recurse ./legal-ai legal-ai-v1:~/ --zone=us-central1-a

# VM → Local
gcloud compute scp legal-ai-v1:~/legal-ai/submission/submission.zip . --zone=us-central1-a
```

## Kiểm tra
```bash
nvidia-smi                                    # GPU status
curl http://localhost:6333/collections/legal_vn  # Qdrant
curl http://localhost:8000/v1/models              # vLLM
```

## Chi phí
```
L4 on-demand: ~$0.70/hr
L4 spot:      ~$0.15/hr
SSD 200GB:    ~$34/month

Nhớ STOP VM khi không dùng!
```