# Benchmark

**English** | [Tiếng Việt](BENCHMARK.md)

Measured on 22–24 Nov 2026 on an Apple M1 Max CPU. Every system was re-run on the same machine and scored with the same
text normaliser (`vnasr.text.normalize_text`, version 1.3: lower case, numbers written as words, punctuation removed), so
the numbers are comparable with each other. They can differ from the numbers in each author's model card by a few tenths
up to 1 point because of different normalisation (e.g. hynt 30M streaming reports 9.72% on GigaSpeech2; we measure 8.87%).

**SyER** (syllable error rate) = (substituted + deleted + inserted syllables) / syllables in the reference. Vietnamese
is written one syllable per space-separated token, so SyER is practically the WER that other authors report.

## Test sets

| Set | Utterances | Duration | For VietNote ASR | Notes |
|---|---|---|---|---|
| [VietMed](https://huggingface.co/datasets/leduckhai/VietMed) test | 3,437 | 6.0 h | **zero-shot** | medical conversation, many accents, 7 recording conditions |
| [GigaSpeech2](https://huggingface.co/datasets/speechcolab/gigaspeech2)-vi test | 4,422 | 11.0 h | **zero-shot** | Vietnamese YouTube videos |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) vi test + [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) test | 1,617 | 3.8 h | train splits used for training | read speech |
| Code-switching dev | 1,600 | 4.6 h | same sources as part of the training data | 4 sources × 400 utterances: Vietnamese–English conversation (YouTube), multi-accent, read, medical with English terms |

hynt's models were trained on VietMed_Labeled, Sub-GigaSpeech2-Vi and FLEURS (per their model cards). The model cards of
VietASR and viet-asr do not state their training data.

## Results (SyER %)

| System | Type | Params | VietMed ⭐ | GigaSpeech2 ⭐ | FLEURS+VIVOS | Dev | English words recognised (dev) |
|---|---|---|---|---|---|---|---|
| **VietNote-Zipformer-78M-Streaming (fp32)** | streaming | 77.7M | **13.87** | 10.65 | **6.88** | **9.62** | **75.5%** |
| **VietNote-Zipformer-78M-Streaming (int8)** | streaming | 77.7M | – | – | 7.28 | – | – |
| VietNote-Zipformer-78M offline (fp32) | offline | 77.7M | 13.44 | 9.67 | 5.69 | 8.64 | 76.5% |
| VietNote-Zipformer-78M offline (int8) | offline | 77.7M | – | – | 6.09 | – | – |
| [hynt Zipformer 30M streaming](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h) | streaming | 30M | 14.30 | **8.87** | 7.73 | 12.58 | 24.7% |
| [Moonshine streaming tiny vi](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny-vi) | streaming | 27M | 19.74 | 14.38 | 10.50 | 22.10 | 20.8% |
| [hynt Zipformer 150M](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) | offline | 153M | 11.90 | 7.20 | 5.14 | 10.81 | 35.5% |
| [hynt Zipformer 30M](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | offline | 30M | 12.93 | 7.40 | 6.06 | 11.01 | 28.9% |
| [viet-asr (dangvansam)](https://huggingface.co/dangvansam/viet-asr) int8 | offline | 86M | 17.33 | 14.71 | 15.21 | 16.52 | 5.0% |
| [VietASR Zipformer 70k h](https://huggingface.co/zzasdf/viet_iter3_pseudo_label) | offline | 68M | 14.19 | 6.75 | 7.50 | 13.22 | 14.3% |
| [PhoWhisper large](https://huggingface.co/vinai/PhoWhisper-large) | offline | 1.5B | 16.74 | – | 6.03 | 16.31 | 30.8% |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | offline | 1.5B | 36.64 | – | 8.08 | 15.21 | 67.8% |

⭐ zero-shot for VietNote ASR. Streaming models are fed 100 ms of audio at a time, as from a microphone. Decoding: modified
beam search with 4 active paths for all transducer models. Whisper/PhoWhisper were not run on GigaSpeech2. The int8
models have only been measured on FLEURS+VIVOS so far.

## Finished text (FLEURS, casing, punctuation and digits scored)

FLEURS keeps the original written sentence (capitals, punctuation, digits), so it can grade the finished output, the way
[viet-asr](https://github.com/dangvansam/viet-asr) reports its results. 857 test sentences, case-sensitive matching,
Unicode NFC (`scripts/score_written.py`).

- **Full WER:** each punctuation mark is a token of its own, so a missing comma counts as an error.
- **WER without punctuation:** casing and the written form of numbers still count.
- **SyER:** normalised as in the tables above; measures only whether the words were heard correctly.

| System | Output | Full WER | WER without punctuation | SyER |
|---|---|---|---|---|
| Soniox (API) | formats itself | **10.09** | **7.08** | **3.58** |
| **VietNote offline + ITN + punctuation** | 3 steps | 12.42 | 9.34 | 5.72 |
| **VietNote streaming + ITN + punctuation** | 3 steps | 13.55 | 10.49 | 6.89 |
| Whisper large-v3 | formats itself | 15.05 | 11.44 | 7.85 |
| Moonshine base vi | formats itself | 17.01 | 13.56 | 9.53 |
| PhoWhisper large | formats itself | 20.57 | 17.17 | 6.48 |
| viet-asr 86M int8 | formats itself | 21.08 | 18.95 | 15.20 |
| hynt Zipformer 30M streaming | lower case, no punctuation | 24.61 | 18.71 | 8.30 |
| hynt Zipformer 150M | UPPER CASE, no punctuation | 100.47 | 102.25 | 5.45 |

viet-asr produces well-formatted text but drops many words, especially foreign names (SyER 15.2%), so it still ranks
behind models that hear the words more accurately. The hynt models hear well (SyER 5.45%) but do not format their output,
so this way of scoring is unfair to them.

## Against commercial models and audio LLMs (400-utterance subsets)

Paid APIs charge per minute, so they were run on subsets only: 400 dev utterances (100 per source, random with seed 0)
and 400 FLEURS+VIVOS utterances (200 + 200). Gemini and GPT-audio were called through OpenRouter (Oct–Nov 2026), Soniox
through its official API. Open models are scored on exactly the same utterances.

**Code-switching dev (400 utterances)**

| System | Type | SyER | English words recognised |
|---|---|---|---|
| VietNote-Zipformer-78M offline | open, CPU | **8.40** | 77.3% |
| Gemini 4 Argon (Google) | API, LLM | 8.62 | **87.7%** |
| **VietNote-Zipformer-78M-Streaming** | open, CPU, streaming | 9.54 | 75.6% |
| Soniox | API, commercial ASR | 10.32 | 78.4% |
| hynt Zipformer 150M | open | 10.93 | 36.1% |
| GPT-audio (OpenAI) | API, LLM | 11.67 | 79.1% |
| Whisper large-v3 | open, 1.5B | 15.03 | 66.7% |
| GPT-audio-mini (OpenAI) | API, LLM | 15.15 | 71.4% |
| Gemini 3.8 Flash (low reasoning) | API, LLM | 16.51 | 81.1% |
| PhoWhisper large | open, 1.5B | 17.58 | 28.4% |

**FLEURS + VIVOS (400 utterances, read speech)**

| System | SyER | FLEURS | VIVOS |
|---|---|---|---|
| Gemini 4 Argon | **2.61** | 2.21 | 3.85 |
| Soniox | 4.04 | 3.61 | 5.38 |
| hynt Zipformer 150M | 5.18 | 5.36 | 4.64 |
| VietNote-Zipformer-78M offline | 5.32 | 5.29 | 5.43 |
| PhoWhisper large | 5.66 | 5.93 | 4.84 |
| **VietNote-Zipformer-78M-Streaming** | 6.45 | 6.48 | 6.37 |
| GPT-audio | 7.24 | 3.37 | 19.15 |
| Whisper large-v3 | 7.99 | 7.76 | 8.69 |
| GPT-audio-mini | 8.47 | 4.87 | 19.55 |
| Gemini 3.8 Flash (low reasoning) | 12.02 | 14.41 | 4.64 |

How to read these tables:
- The dev set comes from the same sources as part of VietNote's training data, so it favours VietNote. The FLEURS+VIVOS
  table is fairer, and there Gemini 4 Argon and Soniox are far ahead.
- LLMs tend to "edit" speech (drop fillers, tighten wording), which is penalised against verbatim labels.
- LLMs have operational failures that dedicated ASR does not: GPT-audio answered like a chatbot instead of transcribing
  on 11/400 FLEURS+VIVOS clips ("Sorry, I can't hear…"), mostly very short VIVOS clips; Gemini 3.8 Flash truncated its
  output on 22/400.
- The APIs have not been run on VietMed and GigaSpeech2.

## By accent and recording condition (VietMed)

| | Utterances | VietNote streaming | hynt 30M streaming | hynt 150M | PhoWhisper |
|---|---|---|---|---|---|
| South-East | 750 | 8.10 | 8.43 | 6.36 | 11.89 |
| North | 693 | 9.83 | 10.22 | 7.92 | 11.10 |
| South-West (Mekong Delta) | 1,164 | 16.18 | 16.57 | 15.01 | 18.80 |
| South-Central | 809 | 18.63 | 19.30 | 15.19 | 22.35 |
| North-Central | 21 | 30.70 | 28.45 | 31.83 | 34.37 |
| Podcast | 766 | 8.90 | 9.00 | 7.98 | 13.55 |
| Telephone | 607 | 9.73 | 10.45 | 7.86 | 11.29 |
| News | 312 | 10.48 | 11.04 | 8.89 | 13.27 |
| Diagnosis | 192 | 9.23 | 9.95 | 6.92 | 10.45 |
| Consultation | 446 | 17.35 | 17.43 | 15.14 | 18.73 |
| Talk show | 730 | 18.25 | 18.42 | 16.69 | 20.13 |
| Lecture | 384 | 20.84 | 22.15 | 15.97 | 27.62 |

## Speed (one CPU core, M1 Max)

RTF = processing time / audio duration (lower is faster). Streaming: fed 100 ms at a time; offline: one whole utterance
at once. Measured on 200 FLEURS utterances with nothing else running.

| Model | RTF | Faster than real time |
|---|---|---|
| VietNote ASR streaming int8 | 0.037 | 27× |
| VietNote ASR streaming fp32 | 0.057 | 17× |
| VietNote-Zipformer-78M offline (int8) | 0.026 | 38× |
| VietNote-Zipformer-78M offline (fp32) | 0.047 | 21× |
| hynt Zipformer 30M streaming (fp16) | 0.037 | 27× |
| hynt Zipformer 30M offline (int8) | 0.018 | 55× |

hynt 30M offline was measured on 60 FLEURS utterances under the same conditions. The hynt model is 2.5× smaller, so it is
faster offline; in streaming mode the two models run at the same speed.

## Scoring your own data

```bash
python scripts/evaluate.py --model models/streaming/fp32 --tsv your_test.tsv
```

`your_test.tsv`: one line per utterance, `audio_path<TAB>reference`. The code-switching dev set cannot be published
because part of its source audio may not be redistributed.

## Known issue with the default int8 export

The int8 model that icefall's ONNX export produces by default (per-tensor quantisation) makes this model stop emitting
text after a few seconds: 82% SyER on FLEURS+VIVOS. The released files use per-channel quantisation
(`scripts/quantize_int8.py`) and are only about 0.4 points behind fp32.
