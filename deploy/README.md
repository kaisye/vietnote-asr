# Triển khai demo lên VM (Google Cloud)

VM CPU, Caddy lo HTTPS (Let's Encrypt), máy chủ chạy bằng systemd. Không cần GPU. Một e2-standard-4 (4 vCPU, 16 GB)
phục vụ 8–10 người nói cùng lúc với độ trễ khoảng 1 giây; 16 người vẫn chạy nhưng chậm vài giây (đo thật).

```bash
bash scripts/download_models.sh                       # models/
gcloud auth login && gcloud config set project <PROJECT_ID>
DOMAIN=asr.example.com bash deploy/create_vm.sh       # IP tĩnh, mở cổng 80/443, tạo VM; in ra IP
# thêm bản ghi DNS: A  asr  <IP>   (Cloudflare: để "DNS only")
bash deploy/bundle.sh && DOMAIN=asr.example.com bash deploy/upload.sh
```

Sửa vùng, cỡ máy, tên miền trong `deploy/config.sh`. Xem log: `gcloud compute ssh asr-demo -- sudo journalctl -u asr-demo -f`.
Máy chủ không lưu âm thanh (không bật `--dump`).
