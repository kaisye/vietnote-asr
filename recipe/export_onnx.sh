#!/usr/bin/env bash
# Export a fine-tuned epoch to streaming ONNX (fp32 + int8) for sherpa-onnx, the same format as the release.
#   bash export_onnx.sh <recipe_dir> <exp_dir> <tokens.txt> <epoch> [avg]
set -euo pipefail
RECIPE=$1; EXP=$2; TOKENS=$3; EPOCH=$4; AVG=${5:-1}
cd "$RECIPE"
python export-onnx-streaming.py --exp-dir "$EXP" --tokens "$TOKENS" --epoch "$EPOCH" --avg "$AVG" --use-averaged-model 0 \
  --num-encoder-layers 2,2,3,4,3,2 --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,384,512,384,256 --decoder-dim 512 --joiner-dim 512 \
  --causal 1 --chunk-size 32 --left-context-frames 128
cp "$TOKENS" "$EXP/"
ls -la "$EXP"/*.onnx
