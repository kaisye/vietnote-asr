# VietNote ASR

**English** | [Tiếng Việt](README.md)

**Streaming Vietnamese–English (code-switching) speech recognition that runs on a CPU.**

Model: **[VietNote-Zipformer-78M-Streaming](https://huggingface.co/Kaisye/VietNote-Zipformer-78M-Streaming)**, a 78M-parameter Zipformer transducer.

[![Demo](https://img.shields.io/badge/demo-asr.vietnote.vn-0f6e56)](https://asr.vietnote.vn)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-Hugging%20Face-yellow)](https://huggingface.co/Kaisye/VietNote-Zipformer-78M-Streaming)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE) [![Weights: CC BY-NC-SA 4.0](https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-lightgrey)](LICENSE-MODEL)

Try it in your browser: **https://asr.vietnote.vn** (speak into the microphone, read a prepared script, or play a sample).

- **Streaming:** text appears while you speak, in steps of 640 ms of audio.
- **Vietnamese mixed with English (code-switching):** recognises about **75% of the English words** in mixed speech,
  2–15× the open Vietnamese models we tested, close to the commercial Soniox API (78%).
- **Finished text:** digits (1,5 tỷ · 250.000 · 15/12 · 0912 345 678 · tỷ số 2-1), punctuation, casing and correctly spelled
  brand names; handles hesitations and self-corrections ("ba… à không, bốn triệu" → 4 triệu).
- **Small and fast on a CPU:** 83 MB (int8), about **27× faster than real time on one CPU core**, no GPU needed. Runs
  with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on Linux, macOS, Windows, Android, iOS and Raspberry Pi.
- **Fine-tunable** on your own data: PyTorch checkpoint and [icefall](https://github.com/k2-fsa/icefall) recipe
  included, see [docs/FINETUNE.en.md](docs/FINETUNE.en.md).

## Results

Syllable error rate (SyER, %, lower is better). Every system was re-run on the same machine and scored with the same
text normaliser. Details, method and caveats: [docs/BENCHMARK.en.md](docs/BENCHMARK.en.md).

| System | Type | VietMed ⭐ | GigaSpeech2 ⭐ | FLEURS+VIVOS | Code-switching dev | English words recognised |
|---|---|---|---|---|---|---|
| **VietNote-Zipformer-78M-Streaming** | streaming, 78M | **13.87** | 10.65 | **6.88** | **9.62** | **75.5%** |
| hynt Zipformer 30M streaming | streaming, 30M | 14.30 | **8.87** | 7.73 | 12.58 | 24.7% |
| Moonshine streaming tiny vi | streaming, 27M | 19.74 | 14.38 | 10.50 | 22.10 | 20.8% |
| hynt Zipformer 150M | offline | 11.90 | 7.20 | 5.14 | 10.81 | 35.5% |
| hynt Zipformer 30M | offline | 12.93 | 7.40 | 6.06 | 11.01 | 28.9% |
| VietASR Zipformer (70k h) | offline | 14.19 | 6.75 | 7.50 | 13.22 | 14.3% |
| viet-asr 86M (int8) | offline | 17.33 | 14.71 | 15.21 | 16.52 | 5.0% |
| PhoWhisper large | offline | 16.74 | – | 6.03 | 16.31 | 30.8% |
| Whisper large-v3 | offline | 36.64 | – | 8.08 | 15.21 | 67.8% |

⭐ Zero-shot for VietNote ASR (no part of the set was used for training). hynt models were trained on VietMed_Labeled,
Sub-GigaSpeech2 and FLEURS (per their model cards). VietNote ASR used the FLEURS and VIVOS train splits; the dev set comes
from the same sources as part of its training data, so it favours VietNote ASR.

**Finished text** (FLEURS test, 857 sentences, scored case-sensitively against the original written sentences with
punctuation marks as tokens): Soniox 10.09 · **VietNote offline + ITN + punctuation 12.42** · **VietNote streaming +
pipeline 13.55** · Whisper large-v3 15.05 · PhoWhisper 20.57 · viet-asr 86M 21.08.

Against commercial APIs and audio LLMs (Gemini, GPT-audio, Soniox) on 400-utterance subsets: on code-switching speech
VietNote streaming scores 9.54 (offline 8.40) vs Gemini 4 Argon 8.62 and Soniox 10.32, but on read speech it is far
behind (6.45 vs 2.61 for Gemini 4 Argon). See [docs/BENCHMARK.en.md](docs/BENCHMARK.en.md).

**To be clear:** larger offline models (hynt 150M) are more accurate on pure Vietnamese, and on GigaSpeech2 we trail the
models trained on it. What sets VietNote ASR apart is streaming, English inside Vietnamese sentences, and finished text.

Still weak: South-Western and Central accents (16–31% SyER on VietMed vs 8–10% for Northern and South-Eastern speakers);
multi-speaker, far-field conversation; rare names and English technical terms.

## Quick start

```bash
git clone https://github.com/kaisye/vietnote-asr && cd vietnote-asr
pip install -e ".[asr]"
bash scripts/download_models.sh streaming/int8      # or no argument to download everything

python examples/transcribe_file.py audio.wav --model models/streaming/int8
```

```
raw : giá vé là hai trăm năm mươi nghìn một người tăng mười hai phần trăm so với năm hai nghìn không trăm hai mươi tư
itn : giá vé là 250.000 một người tăng 12% so với năm 2024
```

Add punctuation and casing (`pip install -e ".[punct]"`, download `punct`):

```bash
python examples/transcribe_file.py audio.wav --punct models/punct --hotwords data/hotwords.txt
# text: Giá vé là 250.000 một người, tăng 12% so với năm 2024.
```

Live captions from the microphone in a terminal: `pip install sounddevice && python examples/microphone.py`.

### In your code

```python
import sherpa_onnx, soundfile as sf
from vnasr.itn import spoken_to_digits

rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder="models/streaming/int8/encoder.int8.onnx",
    decoder="models/streaming/int8/decoder.int8.onnx",
    joiner="models/streaming/int8/joiner.int8.onnx",
    tokens="models/tokens.txt", num_threads=2,
    decoding_method="modified_beam_search", max_active_paths=4)

audio, sr = sf.read("audio.wav", dtype="float32")   # 16 kHz mono
s = rec.create_stream()
for i in range(0, len(audio), 1600):                 # 100 ms at a time, as from a microphone
    s.accept_waveform(sr, audio[i:i + 1600])
    while rec.is_ready(s):
        rec.decode_stream(s)
    print("\r" + rec.get_result(s), end="")
s.input_finished()
while rec.is_ready(s):
    rec.decode_stream(s)
print("\n" + spoken_to_digits(rec.get_result(s).lower()))
```

## Models

| Folder on Hugging Face | Use it for | Size | Speed (1 CPU core) |
|---|---|---|---|
| `streaming/int8` | real-time apps, phones, small CPUs | 83 MB | ~27× real time |
| `streaming/fp32` | real time, most accurate (runs the web demo) | 274 MB | ~17× |
| `offline/int8` | files, no streaming needed | 88 MB | ~38× |
| `offline/fp32` | files, most accurate | 280 MB | ~21× |
| `pytorch/pretrained.pt` | fine-tuning | 311 MB | – |
| `punct/` | punctuation and casing (XLM-RoBERTa base, ONNX; int8 version 206 MB) | 399 MB | 15–35 ms per sentence |

Speeds measured on one core of an Apple M1 Max. All four ASR models share the same weights; the offline export sees the
whole utterance and is about 1 SyER point more accurate.

## Finished text

The model outputs lower case with numbers as words. Three steps turn it into text ready to use; the demo runs them in
this order:

```
ASR → ① ITN (numbers) → ② punctuation + casing → ③ hotwords (names)
```

**① ITN** ([src/vnasr/itn.py](src/vnasr/itn.py)): hand-written rules, no parameters, about 1.5 ms per sentence.

```python
from vnasr.itn import spoken_to_digits
spoken_to_digits("chuyển khoản cho chị ba à không bốn triệu rưỡi nhé")   # 'chuyển khoản cho chị 4,5 triệu nhé'
spoken_to_digits("hẹn anh bảy giờ ba mươi ngày mười lăm tháng mười hai")  # 'hẹn anh 7 giờ 30 ngày 15 tháng 12'
spoken_to_digits("số điện thoại không chín một hai ba bốn năm sáu bảy tám")  # '... 0912345678'
spoken_to_digits("sau khi đạt tỷ số sáu sáu")                              # 'sau khi đạt tỷ số 6-6'
```

Checked on [VietAI spoken_norm](https://huggingface.co/datasets/VietAI/spoken_norm_assignment): 78.7% of numbers get
the right value and only 0.26% of ordinary words are wrongly converted; on news bulletins with exact transcripts, 100%
of numbers above 10 are right. Robust to fillers ("ừm", "à") and self-corrections.

**② Punctuation and casing** ([punct/](punct/)): XLM-RoBERTa base with two classification heads (punctuation: none /
comma / period / question mark; casing: lower / Capitalised / UPPER / mixed such as iPhone). Casing 97% correct; F1
period 0.73, question mark 0.67, comma 0.56. The vocabulary is pruned from 250k to 24k Vietnamese and English tokens
(`punct/prune_vocab.py`): 1.1 GB becomes 399 MB with identical output. `model.int8.onnx` (206 MB, twice as fast, add
`--punct-int8` to the demo) is within 0.006 F1 of fp32.

**③ Hotwords: proper names** ([src/vnasr/hotwords.py](src/vnasr/hotwords.py), [data/hotwords.txt](data/hotwords.txt)):
names with their Vietnamese-style readings, e.g. `VinFast | vin phát`, `Nvidia | en vi đi a`. Names are boosted during
decoding and readings are replaced by the correct spelling. About 70 well-known brands in Vietnam are included; add your
own to the file.

## Web demo

```bash
pip install -e ".[demo]"
bash scripts/download_models.sh
python demo/server.py --model-dir models/streaming/fp32 --punct-dir models/punct \
    --vad models/silero_vad.onnx --hotwords data/hotwords.txt --bpe-model models/bpe.model
# open http://localhost:8765
```

The server does not store audio unless `--dump` is set. Deploying to a VM (Caddy HTTPS, systemd): [deploy/](deploy/).
A 4-vCPU VM comfortably serves 8–10 simultaneous speakers (measured on Google Cloud e2-standard-4).

## Model card

- **Name:** VietNote-Zipformer-78M-Streaming
- **Architecture:** causal Zipformer2 (size M, 77.7M parameters), stateless transducer, pruned RNN-T loss.
- **Tokens:** BPE 6,144, shared by Vietnamese and English.
- **Streaming:** 32-frame chunks (640 ms), 128 frames of left context; exported to ONNX for sherpa-onnx.
- **Training data:** about 1,500 hours: public Vietnamese speech datasets, additionally collected Vietnamese–English
  code-switching speech, synthetic (TTS) speech and a portion of data that cannot be shared; part of the labels were
  produced automatically. The FLEURS and VIVOS train splits are included. **No part of VietMed or GigaSpeech2 was used.**
- **Training:** [icefall](https://github.com/k2-fsa/icefall), one RTX 5060 Ti 16 GB GPU, bf16, 45 epochs, average of 2 checkpoints.

## Repository layout

```
src/vnasr/      ITN, hotwords, text normalisation, scoring (SyER/CER)
examples/       file and microphone transcription
demo/           web server + UI (asr.vietnote.vn)
punct/          punctuation model: inference, ONNX export, vocabulary pruning, training
recipe/         icefall recipe setup, data preparation, fine-tuning, ONNX export
scripts/        model download, int8 quantisation, evaluation
deploy/         VM deployment
tests/          ~1,500 tests for ITN, hotwords, normalisation
```

Run the tests: `pip install -e ".[dev]" && pytest`.

## License and credits

- **Code:** [Apache-2.0](LICENSE).
- **Weights** (ASR models, PyTorch checkpoint, punctuation model): [CC BY-NC-SA 4.0](LICENSE-MODEL), **non-commercial
  use only**; fine-tuned or converted weights must keep the same licence. Reason: part of the training data (VIVOS,
  Bud500) is licensed CC BY-NC-SA. For commercial use, get in touch via
  [Issues](https://github.com/kaisye/vietnote-asr/issues). Weights downloaded before 25 Nov 2026 were released under
  Apache-2.0.

The punctuation model is fine-tuned from
[XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base) (MIT). Demo audio samples are from
[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0). Built on [k2](https://github.com/k2-fsa/k2),
[icefall](https://github.com/k2-fsa/icefall), [lhotse](https://github.com/lhotse-speech/lhotse) and
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Thanks to the authors of the datasets and of the models we compare
against.

### How to cite

If you use, fine-tune or redistribute VietNote ASR (including fine-tuned, quantised or converted weights), please:
- keep the [NOTICE](NOTICE), [LICENSE](LICENSE) and [LICENSE-MODEL](LICENSE-MODEL) files with it;
- state in your README or model card: *"Based on VietNote ASR (https://github.com/kaisye/vietnote-asr)"*;
- cite it in papers or reports:

```bibtex
@misc{vietnote_asr_2026,
  author       = {Kaisye},
  title        = {{VietNote ASR}: Streaming {Vietnamese}--{English} Code-Switching Speech Recognition},
  year         = {2026},
  howpublished = {\url{https://github.com/kaisye/vietnote-asr}},
  note         = {Model: VietNote-Zipformer-78M-Streaming, \url{https://huggingface.co/Kaisye/VietNote-Zipformer-78M-Streaming}}
}
```
