"""Live captions from your microphone in the terminal (streaming model, ITN on each finished segment).

    pip install sherpa-onnx sounddevice numpy
    python examples/microphone.py [--model models/streaming/int8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vnasr.itn import spoken_to_digits  # noqa: E402


def main():
    import sherpa_onnx
    import sounddevice as sd

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/streaming/fp32")
    p.add_argument("--tokens", type=Path, default=ROOT / "models/tokens.txt")
    a = p.parse_args()
    pick = lambda part: str(next(a.model.glob(f"{part}*.onnx")))  # noqa: E731
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"), tokens=str(a.tokens),
        num_threads=2, decoding_method="modified_beam_search", max_active_paths=4, enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.0, rule2_min_trailing_silence=0.8, rule3_min_utterance_length=15)
    s = rec.create_stream()
    print("Nói đi (Ctrl+C để dừng)...")
    last = ""
    with sd.InputStream(channels=1, samplerate=16000, dtype="float32", blocksize=1600) as mic:
        while True:
            block, _ = mic.read(1600)
            s.accept_waveform(16000, block[:, 0])
            while rec.is_ready(s):
                rec.decode_stream(s)
            text = rec.get_result(s).strip().lower()
            if rec.is_endpoint(s):
                if text:
                    print("\r" + spoken_to_digits(text) + " " * 20)
                rec.reset(s)
                last = ""
            elif text != last:
                print("\r" + text[-100:], end="", flush=True)
                last = text


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
