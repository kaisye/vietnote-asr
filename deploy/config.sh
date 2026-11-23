# Settings shared by the deploy scripts. Edit here, nothing else needs to change.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION=asia-southeast1          # Singapore: lowest latency from Vietnam
ZONE=asia-southeast1-b
VM=asr-demo
MACHINE=e2-standard-4           # 4 vCPU, 16 GB
DISK_GB=30
DOMAIN=${DOMAIN:-asr.example.com}   # your domain; add an A record to the VM IP
EMAIL_FOR_TLS=""                # optional: Let's Encrypt expiry notices
REMOTE_DIR=/opt/asr
