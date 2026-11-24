"""Score finished text (casing, punctuation, digits) against FLEURS written-form references (2026-11-24).

FLEURS keeps the original sentence (capitals, punctuation, digits), so it can grade the whole output, the way
viet-asr's README does. Three numbers per system, all case-sensitive Unicode NFC:
  * WER-full:  punctuation marks count as their own tokens (",", ".", "?" ...), so a missing comma is an error;
  * WER-cased: punctuation dropped, casing and the written form of numbers still count;
  * SyER:      our usual score (vnasr.text.normalize_text on both sides), for reference.
Systems listed with --pipeline get our post-processing first: ITN (spoken_to_digits), the punctuation/casing model,
hotwords; the others are scored on their raw output.

    python scripts/bench/score_written.py --refs data/ext/fleurs/refs.jsonl --hyps data/ext/all/hyps \
        --punct-dir exp/punct_small --pipeline ours-avg45-streaming ours-avg45-offline
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from vnasr.text import normalize_text

ROOT = Path(__file__).resolve().parents[2]
PUNCT = re.compile(r"([,.?!;:…\"“”()\[\]])")


def tokens(text: str, punct: bool) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    text = PUNCT.sub(r" \1 " if punct else " ", text)
    # "-" between words or alone is punctuation; inside a word (Covid-19, 15-10) it stays
    text = re.sub(r"(?<!\w)-|-(?!\w)", " - " if punct else " ", text)
    return text.split()


def edits(r: list[str], h: list[str]) -> int:
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[-1]


def wer(pairs, fn) -> float:
    e = n = 0
    for r, h in pairs:
        rt, ht = fn(r), fn(h)
        e += edits(rt, ht)
        n += len(rt)
    return 100 * e / max(1, n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--hyps", type=Path, required=True)
    p.add_argument("--systems", nargs="*", help="default: every hyps/*.jsonl")
    p.add_argument("--pipeline", nargs="*", default=[], help="systems that get ITN + punctuation + hotwords")
    p.add_argument("--punct-dir", type=Path)
    p.add_argument("--punct-int8", action="store_true")
    p.add_argument("--hotwords", type=Path, default=ROOT / "data/voice/hotwords.txt")
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    refs = {r["id"]: r["text"] for r in map(json.loads, a.refs.open(encoding="utf-8"))}
    finish = None
    if a.pipeline:
        sys.path.insert(0, str(ROOT / "scripts/punct"))
        from punctuate import Punctuator
        from vnasr.hotwords import load as load_hot
        from vnasr.itn import spoken_to_digits
        punct = Punctuator(a.punct_dir, onnx=True, threads=4, int8=a.punct_int8)
        hot = load_hot(a.hotwords)

        def finish(t):
            t = t.strip().lower()
            return hot.apply(punct(spoken_to_digits(t))) if t else ""

    systems = a.systems or sorted(f.stem for f in a.hyps.glob("*.jsonl"))
    results = {}
    print(f"{'system':28} {'rows':>5} {'WER-full':>9} {'WER-cased':>10} {'SyER':>6}")
    for s in systems:
        f = a.hyps / f"{s}.jsonl"
        if not f.exists():
            continue
        hyps = {r["id"]: r["text"] for r in map(json.loads, f.open(encoding="utf-8"))}
        names = [s] + ([f"{s}+pipeline"] if s in a.pipeline else [])
        for name in names:
            pairs = []
            for rid, ref in refs.items():
                h = hyps.get(rid, "")
                pairs.append((ref, finish(h) if name.endswith("+pipeline") else h))
            res = {"rows": sum(rid in hyps for rid in refs),
                   "wer_full": wer(pairs, lambda t: tokens(t, True)),
                   "wer_cased": wer(pairs, lambda t: tokens(t, False)),
                   "syer": wer(pairs, lambda t: normalize_text(t).split())}
            results[name] = res
            print(f"{name:28} {res['rows']:5d} {res['wer_full']:9.2f} {res['wer_cased']:10.2f} {res['syer']:6.2f}",
                  flush=True)
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
