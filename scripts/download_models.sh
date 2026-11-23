#!/usr/bin/env bash
# Download the released models from Hugging Face into ./models (about 2 GB with everything; use --include to pick).
#   bash scripts/download_models.sh                 # everything
#   bash scripts/download_models.sh streaming/int8  # one folder
set -euo pipefail
REPO=${HF_REPO:-Kaisye/vietnote-asr}
cd "$(dirname "$0")/.."
command -v hf >/dev/null || python3 -m pip install -q "huggingface_hub>=0.26"
if [[ $# -gt 0 ]]; then
  hf download "$REPO" --local-dir models --include "$1/*" --include "tokens.txt" --include "bpe.model"
else
  hf download "$REPO" --local-dir models
fi
# Silero VAD (MIT) for the demo's speech gate.
[[ -f models/silero_vad.onnx ]] || curl -sL -o models/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
ls models
