"""Syllable error rate (SyER) of a model on your own test set, scored like the benchmark in the README.

    python scripts/evaluate.py --model models/streaming/fp32 --tsv test.tsv [--offline] [--int8]

The TSV has `audio_path<TAB>transcript`. Reference and hypothesis are both normalised with vnasr.text.normalize_text
(lower case, numbers as words, no punctuation) before alignment.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vnasr.metrics import score  # noqa: E402


def main():
    import sherpa_onnx

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--tokens", type=Path, help="default <model>/tokens.txt, else models/tokens.txt")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    tokens = a.tokens or (a.model / "tokens.txt" if (a.model / "tokens.txt").exists() else ROOT / "models/tokens.txt")

    def pick(part):
        files = [f for f in sorted(a.model.glob(f"{part}*.onnx")) if (".int8." in f.name) == a.int8]
        return str(files[0])

    kw = dict(encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"), tokens=str(tokens),
              num_threads=a.threads, decoding_method="modified_beam_search", max_active_paths=4)
    rec = (sherpa_onnx.OfflineRecognizer if a.offline else sherpa_onnx.OnlineRecognizer).from_transducer(**kw)
    pairs, dur, comp = [], 0.0, 0.0
    for line in a.tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path, _, ref = line.partition("\t")
        audio, sr = sf.read(path.strip(), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            audio = np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)
        dur += len(audio) / 16000
        t = time.perf_counter()
        s = rec.create_stream()
        if a.offline:
            s.accept_waveform(16000, audio)
            rec.decode_stream(s)
            hyp = s.result.text
        else:
            s.accept_waveform(16000, np.concatenate([audio, np.zeros(int(0.66 * 16000), np.float32)]))
            s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            hyp = rec.get_result(s)
        comp += time.perf_counter() - t
        pairs.append((ref, hyp.lower()))
    r = score(pairs)
    print(f"{len(pairs)} utterances, {dur / 60:.1f} min: SyER {r.syer:.2%}  CER {r.cer:.2%}  RTF {comp / dur:.3f}")


if __name__ == "__main__":
    main()
