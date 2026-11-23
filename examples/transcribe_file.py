"""Transcribe an audio file with VietNote ASR (sherpa-onnx), then turn the text into finished text.

    pip install sherpa-onnx soundfile numpy
    python examples/transcribe_file.py audio.wav                         # streaming fp32 (as on the web demo)
    python examples/transcribe_file.py audio.wav --model models/offline/int8 --offline
    python examples/transcribe_file.py audio.wav --punct models/punct    # + punctuation/casing (needs onnxruntime, transformers)

Output: the raw model text (lower case, numbers as words), then the ITN text (digits), then punctuated text.
Long files are cut into 20 s pieces for the offline model; the streaming model takes any length.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vnasr.hotwords import load as load_hotwords  # noqa: E402
from vnasr.itn import spoken_to_digits  # noqa: E402


def pick(model_dir: Path, part: str) -> str:
    return str(next(model_dir.glob(f"{part}*.onnx")))


def read_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:  # simple linear resampling; use ffmpeg/sox for best quality
        audio = np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def main():
    import sherpa_onnx

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("--model", type=Path, default=ROOT / "models/streaming/fp32")
    p.add_argument("--offline", action="store_true", help="the model dir holds an offline (non-streaming) export")
    p.add_argument("--tokens", type=Path, default=ROOT / "models/tokens.txt")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--hotwords", type=Path, help="e.g. data/hotwords.txt; needs --bpe-model")
    p.add_argument("--bpe-model", type=Path, default=ROOT / "models/bpe.model")
    p.add_argument("--punct", type=Path, help="models/punct: add punctuation and casing")
    a = p.parse_args()

    kw = dict(encoder=pick(a.model, "encoder"), decoder=pick(a.model, "decoder"), joiner=pick(a.model, "joiner"),
              tokens=str(a.tokens), num_threads=a.threads, decoding_method="modified_beam_search", max_active_paths=4)
    hw = None
    if a.hotwords:
        hw = load_hotwords(a.hotwords)
        out = Path("/tmp/vietnote-hotwords")
        from vnasr.hotwords import write_sherpa_files
        words, vocab = write_sherpa_files(hw, a.bpe_model, out)
        kw.update(hotwords_file=str(words), hotwords_score=2.0, modeling_unit="bpe", bpe_vocab=str(vocab))

    audio = read_audio(a.audio)
    if a.offline:
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)
        parts = []
        for i in range(0, len(audio), 20 * 16000):
            s = rec.create_stream()
            s.accept_waveform(16000, audio[i:i + 20 * 16000])
            rec.decode_stream(s)
            parts.append(s.result.text)
        raw = " ".join(parts)
    else:
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(**kw)
        s = rec.create_stream()
        s.accept_waveform(16000, np.concatenate([audio, np.zeros(int(0.66 * 16000), np.float32)]))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        raw = rec.get_result(s)
    raw = raw.strip().lower()
    text = spoken_to_digits(raw)
    if hw:
        text = hw.apply(text)
    print("raw :", raw)
    print("itn :", text)
    if a.punct:
        sys.path.insert(0, str(ROOT / "punct"))
        from punctuate import Punctuator
        final = Punctuator(a.punct, onnx=True)(text)
        print("text:", hw.apply(final) if hw else final)


if __name__ == "__main__":
    main()
