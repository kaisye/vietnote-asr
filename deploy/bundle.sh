#!/usr/bin/env bash
# Collect code + models (streaming fp32, punctuation, VAD) into deploy/build/asr.tar for upload.sh.
# Run scripts/download_models.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
B=deploy/build/asr; rm -rf deploy/build; mkdir -p "$B"/models
for f in models/streaming/fp32/encoder.onnx models/punct/model.onnx models/silero_vad.onnx models/bpe.model; do
  [[ -f $f ]] || { echo "missing $f: run scripts/download_models.sh"; exit 1; }
done
cp -R src demo punct data "$B/"
cp -R models/streaming/fp32 "$B/models/asr"; cp -R models/punct "$B/models/punct"
cp models/silero_vad.onnx models/bpe.model "$B/models/"
cp deploy/requirements.txt deploy/asr-demo.service deploy/Caddyfile deploy/setup_vm.sh deploy/config.sh "$B/"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
tar -C deploy/build -cf deploy/build/asr.tar asr
du -sh deploy/build/asr.tar
