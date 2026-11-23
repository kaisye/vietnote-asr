#!/usr/bin/env bash
# Create the static IP, firewall rules and VM. Run once from your machine after `gcloud auth login`.
set -euo pipefail
cd "$(dirname "$0")"; source config.sh
[ -n "$PROJECT" ] || { echo "set a project: gcloud config set project <id>"; exit 1; }
echo "project=$PROJECT zone=$ZONE vm=$VM ($MACHINE)"

gcloud compute addresses describe "$VM-ip" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud compute addresses create "$VM-ip" --region "$REGION" --project "$PROJECT"
IP=$(gcloud compute addresses describe "$VM-ip" --region "$REGION" --project "$PROJECT" --format='value(address)')

gcloud compute firewall-rules describe allow-http-https --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud compute firewall-rules create allow-http-https --project "$PROJECT" \
    --allow tcp:80,tcp:443 --target-tags web --description "Caddy for $DOMAIN"

gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
  --machine-type "$MACHINE" --address "$IP" --tags web \
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
  --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced

echo
echo "VM IP: $IP"
echo "Now add the DNS record at your registrar:  A   asr   $IP   (TTL 300)"
echo "Then: ./bundle.sh && ./upload.sh"
