#!/usr/bin/env bash
# Copy the bundle to the VM and run the setup there. Safe to rerun after a code or model change.
set -euo pipefail
cd "$(dirname "$0")"; source config.sh
gcloud compute scp build/asr.tar "$VM":/tmp/asr.tar --zone "$ZONE" --project "$PROJECT"
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command \
  "sudo rm -rf /tmp/asr && tar -C /tmp -xf /tmp/asr.tar && sudo bash /tmp/asr/setup_vm.sh"
echo "Open https://$DOMAIN (the certificate appears once DNS points to the VM)."
