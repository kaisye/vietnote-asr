#!/usr/bin/env bash
# Fine-tune VietNote ASR on your own data (see docs/FINETUNE.md).
#   bash finetune.sh <recipe_dir> <cuts_dir> <pretrained.pt> <bpe.model> <exp_dir> [num_epochs]
# The model flags must match the released checkpoint; do not change them.
set -euo pipefail
RECIPE=$1; CUTS=$2; CKPT=$3; BPE=$4; EXP=$5; EPOCHS=${6:-10}
cd "$RECIPE"
python finetune.py \
  --world-size 1 --num-epochs "$EPOCHS" --start-epoch 1 --use-fp16 0 \
  --exp-dir "$EXP" --bpe-model "$BPE" --manifest-dir "$CUTS" \
  --do-finetune 1 --finetune-ckpt "$CKPT" --use-mux 0 \
  --num-encoder-layers 2,2,3,4,3,2 --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,384,512,384,256 --decoder-dim 512 --joiner-dim 512 \
  --causal 1 --chunk-size 16,32,64,-1 --left-context-frames 64,128,256,-1 \
  --use-transducer 1 --use-ctc 0 \
  --base-lr 0.0045 --lr-batches 100000 --lr-epochs 100 \
  --max-duration "${MAX_DURATION:-300}" --on-the-fly-feats 1 --enable-musan 0 \
  --bucketing-sampler 1 --num-buckets 10 --num-workers 4 --save-every-n 500 --keep-last-k 5
