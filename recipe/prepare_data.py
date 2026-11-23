"""Turn your recordings and transcripts into lhotse cuts for fine-tuning.

Input: a TSV with one utterance per line, `audio_path<TAB>transcript` (any format soundfile/ffmpeg reads; it is
resampled to 16 kHz mono on the fly). Transcripts can be written normally ("Giá 250.000 đồng, tăng 12%."): they are
normalised exactly like the training labels (lower case, numbers read as words, punctuation removed) with
vnasr.text.normalize_text, so the model keeps producing spoken-form text that the ITN turns back into digits.

    python prepare_data.py --tsv my_data.tsv --out data/my_cuts --valid-fraction 0.05

Writes <out>/train.jsonl.gz and <out>/valid.jsonl.gz.
"""

import argparse
import random
import sys
from pathlib import Path

from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vnasr.text import normalize_text  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--valid-fraction", type=float, default=0.05)
    p.add_argument("--min-sec", type=float, default=1.0)
    p.add_argument("--max-sec", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    recs, sups, skipped = [], [], 0
    for n, line in enumerate(a.tsv.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        path, _, text = line.partition("\t")
        text = normalize_text(text)
        if not text:
            skipped += 1
            continue
        rid = f"ft-{n:07d}"
        rec = Recording.from_file(path.strip(), recording_id=rid)
        if not a.min_sec <= rec.duration <= a.max_sec:
            skipped += 1
            continue
        recs.append(rec)
        sups.append(SupervisionSegment(id=rid, recording_id=rid, start=0.0, duration=rec.duration,
                                       channel=0, text=text, language="vi"))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(recs),
                                 supervisions=SupervisionSet.from_segments(sups))
    cuts = cuts.resample(16000)
    ids = sorted(c.id for c in cuts)
    random.Random(a.seed).shuffle(ids)
    n_valid = max(1, int(len(ids) * a.valid_fraction))
    valid = set(ids[:n_valid])
    a.out.mkdir(parents=True, exist_ok=True)
    cuts.filter(lambda c: c.id not in valid).to_file(a.out / "train.jsonl.gz")
    cuts.filter(lambda c: c.id in valid).to_file(a.out / "valid.jsonl.gz")
    hours = sum(c.duration for c in cuts) / 3600
    print(f"{len(ids)} utterances ({hours:.2f} h): {len(ids) - n_valid} train, {n_valid} valid; skipped {skipped}")


if __name__ == "__main__":
    main()
