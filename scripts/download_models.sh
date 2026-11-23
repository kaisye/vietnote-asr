#!/usr/bin/env bash
# Download the released models from Hugging Face into ./models (about 2 GB with everything; use --include to pick).
#   bash scripts/download_models.sh                 # everything
#   bash scripts/download_models.sh streaming/int8  # one folder
set -euo pipefail
REPO=${HF_REPO:-kaisye/vietnote-asr}
cd "$(dirname "$0")/.."
pip show huggingface_hub >/dev/null 2>&1 || pip install -q "huggingface_hub>=0.26"
if [[ $# -gt 0 ]]; then
  hf download "$REPO" --local-dir models --include "$1/*" "tokens.txt" "bpe.model"
else
  hf download "$REPO" --local-dir models
fi
# Silero VAD (MIT) for the demo's speech gate.
[[ -f models/silero_vad.onnx ]] || curl -sL -o models/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
ls models
