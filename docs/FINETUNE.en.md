# Fine-tuning VietNote ASR on your own data

**English** | [Tiếng Việt](FINETUNE.md)

When to fine-tune: the model needs to hear a particular **regional accent**, a **domain** (medical, legal, call
centre…), **names or terms** that come up often, or a specific **recording environment** better. For a handful of names,
try [hotwords](../README.en.md#finished-text) first; no training needed.

The procedure below has been run end to end (data preparation → fine-tuning → ONNX export → decoding with sherpa-onnx).

## 1. Requirements

| | |
|---|---|
| GPU | NVIDIA with ≥ 12 GB (tested on an RTX 5060 Ti 16 GB). CPU-only works but is very slow. |
| Software | Python 3.11–3.12, PyTorch + [k2](https://k2-fsa.github.io/k2/), [lhotse](https://github.com/lhotse-speech/lhotse), [icefall](https://github.com/k2-fsa/icefall) |
| Checkpoint | `pytorch/pretrained.pt` plus `bpe.model` and `tokens.txt` from Hugging Face |
| Data | At least 1–2 hours with accurate transcripts; 10–50 hours for a clear change. Utterances of 1–20 seconds. |

Install icefall following the [official guide](https://k2-fsa.github.io/icefall/installation/index.html), then pin the
commit we tested:

```bash
git clone https://github.com/k2-fsa/icefall && cd icefall
git checkout 9c966762e058bd7ba79d9da71e42fc06b64b703e
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..
```

Download the checkpoint:

```bash
bash scripts/download_models.sh pytorch      # -> models/pytorch/pretrained.pt, models/bpe.model, models/tokens.txt
```

## 2. Create the recipe

`recipe/make_recipe.py` builds a recipe directory from icefall: the model code is linked unchanged and only the data
loading is patched to use your data (features computed on the fly during training, with random 0.9/1.1 speed
perturbation).

```bash
python recipe/make_recipe.py --icefall /path/to/icefall --out zipformer_vi
```

## 3. Prepare the data

Write a TSV file with one utterance per line: `audio_path<TAB>transcript`.

```
/data/my/0001.wav	Khách hàng chuyển khoản 2,5 triệu cho Vietcombank lúc 9 giờ sáng.
/data/my/0002.wav	Bệnh nhân dùng Paracetamol 500 mg, ngày 3 lần.
```

```bash
python recipe/prepare_data.py --tsv my_data.tsv --out data/my_cuts --valid-fraction 0.05
```

**Important:** transcripts are normalised exactly like the training labels (`vnasr.text.normalize_text`): lower case,
numbers written as words ("2,5 triệu" → "hai phẩy năm triệu"), punctuation removed. Write transcripts normally; do not
normalise them some other way yourself, or the model will learn two spellings. Audio at any sample rate is converted to
16 kHz mono.

Tips:
- Mix in some general data (e.g. Vietnamese [FLEURS](https://huggingface.co/datasets/google/fleurs) or
  [Common Voice](https://commonvoice.mozilla.org)), about 20–30% of the hours, so the model does not forget general
  Vietnamese and English. Just add the lines to the same TSV file.
- Keep a separate test set (not in the TSV) to compare before and after.

## 4. Fine-tune

```bash
bash recipe/finetune.sh $PWD/zipformer_vi $PWD/data/my_cuts \
    $PWD/models/pytorch/pretrained.pt $PWD/models/bpe.model $PWD/exp/my_ft 10
```

Settings in `finetune.sh`:
- The model configuration (`--num-encoder-layers`, `--encoder-dim`, `--causal 1`, `--chunk-size`…) **must stay as it
  is**: it is the architecture of the checkpoint.
- `--base-lr 0.0045 --lr-batches 100000 --lr-epochs 100`: a low learning rate that barely decays, as icefall recommends
  for fine-tuning. With little data (under 5 hours), lower it to `0.002` and train 3–5 epochs.
- `MAX_DURATION=300` (seconds of audio per batch) fits in 16 GB; on a 12 GB GPU use `MAX_DURATION=200`.
- Start again from the original model rather than fine-tuning in several rounds: watch `validation loss` in
  `exp/my_ft/log` and stop when it no longer goes down.

Time (RTX 5060 Ti, measured): about 6 seconds per hour of data per epoch, plus about 1 minute of start-up. 10 hours of
data × 10 epochs ≈ 15 minutes; 100 hours × 10 epochs ≈ 2 hours.

## 5. Export to ONNX and use it

```bash
bash recipe/export_onnx.sh $PWD/zipformer_vi $PWD/exp/my_ft $PWD/models/tokens.txt 10 3   # epoch 10, average of 3 epochs
```

Averaging the last few epochs (`avg` 2–4) is usually better than a single epoch. The result is encoder/decoder/joiner
`.onnx` (fp32) and `.int8.onnx` files, used exactly like the released model:

```bash
python examples/transcribe_file.py test.wav --model exp/my_ft
```

**int8:** the int8 files that icefall produces by default can make a streaming model stop emitting text after a few
seconds. Re-quantise from the fp32 files per channel, as for the released model:

```bash
python scripts/quantize_int8.py exp/my_ft
```

## 6. Compare before and after

```bash
python scripts/evaluate.py --model exp/my_ft --tsv my_test.tsv              # SyER of the fine-tuned model
python scripts/evaluate.py --model models/streaming/fp32 --tsv my_test.tsv  # original model
```

SyER (syllable error rate) is computed on normalised text on both sides, as in the benchmark tables in the README.

## FAQ

- **Do I need to add new words to a dictionary?** No. The 6,144 BPE tokens are shared by Vietnamese and English and can
  spell any word. The word only needs to appear in your fine-tuning transcripts.
- **The model forgot English after fine-tuning?** Mix in code-switching data, or lower the number of epochs or the
  learning rate.
- **Want an offline model (more accurate when you do not need streaming)?** Use
  `export-streaming-as-non-streaming-onnx.py` in the recipe directory with `--chunk-size -1 --left-context-frames -1`.
