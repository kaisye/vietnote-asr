"""Create the VietNote ASR recipe directory (train, fine-tune, decode) from a pinned icefall checkout.

Tested with icefall commit 9c966762e058bd7ba79d9da71e42fc06b64b703e (2026-10-28).

The model code is symlinked unchanged. Only two files are generated, by applying
small text patches to icefall's own files. Each patch asserts that its anchor text
exists exactly once, so a different icefall commit fails loudly instead of silently.
"""

import argparse
import shutil
from pathlib import Path


def patch(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"patch anchor not found exactly once:\n{old[:120]}"
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    asr = args.icefall / "egs/librispeech/ASR"
    src = asr / "zipformer"
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    for file in src.glob("*.py"):
        # Generated files must not be symlinks, or writing them would change icefall itself.
        if file.name not in {"train.py", "finetune.py", "decode.py", "asr_datamodule.py"}:
            (args.out / file.name).symlink_to(file.resolve())

    # Data module: lhotse cuts from our JSONL, on-the-fly features, random speed perturbation.
    dm = (asr / "tdnn_lstm_ctc/asr_datamodule.py").read_text()
    dm = patch(dm, "class LibriSpeechAsrDataModule", "class ViAsrDataModule")
    dm = patch(
        dm,
        "    def train_clean_5_cuts(self) -> CutSet:",
        '''    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        return CutSet.from_file(self.args.manifest_dir / "train.jsonl.gz")

    def valid_cuts(self) -> CutSet:
        logging.info("About to get valid cuts")
        return CutSet.from_file(self.args.manifest_dir / "valid.jsonl.gz")

    def eval_cuts(self, name: str) -> CutSet:
        return CutSet.from_file(self.args.manifest_dir / f"{name}.jsonl.gz")

    def train_clean_5_cuts(self) -> CutSet:''',
    )
    dm = patch(
        dm,
        "            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa",
        "            transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)] + transforms",
    )
    assert "PerturbSpeed" in dm.split("class ")[0] or "import" in dm
    if "PerturbSpeed," not in dm and "PerturbSpeed\n" not in dm.split("class ")[0]:
        dm = patch(dm, "from lhotse.dataset import (", "from lhotse.dataset import (\n    PerturbSpeed,")
    # MUSAN noise is not available here, so it must not be on by default.
    dm = patch(
        dm,
        '            "--enable-musan",\n            type=str2bool,\n            default=True,',
        '            "--enable-musan",\n            type=str2bool,\n            default=False,',
    )
    (args.out / "vi_datamodule.py").write_text(dm)

    # Training script: use our data module and read frame counts from durations.
    train = (src / "train.py").read_text()
    train = patch(
        train,
        "from asr_datamodule import LibriSpeechAsrDataModule",
        "from vi_datamodule import ViAsrDataModule as LibriSpeechAsrDataModule",
    )
    start = train.index("    if params.full_libri:")
    end = train.index("    def remove_short_and_long_utt(c: Cut):")
    train = train[:start] + "    train_cuts = librispeech.train_cuts()\n\n" + train[end:]
    train = patch(
        train,
        "        T = ((c.num_frames - 7) // 2 + 1) // 2",
        "        # Features are computed on the fly, so c.num_frames is None; 100 frames per second.\n"
        "        T = ((int(c.duration * 100) - 7) // 2 + 1) // 2",
    )
    train = patch(
        train,
        "    valid_cuts = librispeech.dev_clean_cuts()\n    valid_cuts += librispeech.dev_other_cuts()\n",
        "    valid_cuts = librispeech.valid_cuts()\n",
    )
    (args.out / "train.py").write_text(train)

    # Fine-tuning script: start from pretrained.pt (--do-finetune 1 --finetune-ckpt) on your own cuts.
    ft = (src / "finetune.py").read_text()
    ft = patch(
        ft,
        "from asr_datamodule import LibriSpeechAsrDataModule",
        "from vi_datamodule import ViAsrDataModule as LibriSpeechAsrDataModule",
    )
    start = ft.index("    gigaspeech_cuts = librispeech.gigaspeech_subset_small_cuts()")
    end = ft.index("    logging.info(train_cuts)\n", start)
    ft = ft[:start] + "    train_cuts = librispeech.train_cuts()\n" + ft[end:]
    ft = patch(
        ft,
        "        T = ((c.num_frames - 7) // 2 + 1) // 2",
        "        # Features are computed on the fly, so c.num_frames is None; 100 frames per second.\n"
        "        T = ((int(c.duration * 100) - 7) // 2 + 1) // 2",
    )
    start = ft.index("    valid_cuts = librispeech.dev_clean_cuts()")
    end = ft.index("    ]\n", ft.index("    valid_dls = [", start)) + len("    ]\n")
    ft = (
        ft[:start]
        + "    valid_sets = [\"valid\"]\n"
        + "    valid_dls = [librispeech.valid_dataloaders(librispeech.valid_cuts())]\n"
        + ft[end:]
    )
    (args.out / "finetune.py").write_text(ft)

    # Decoding script: score any of our cut sets by name instead of LibriSpeech test sets.
    decode = (src / "decode.py").read_text()
    decode = patch(
        decode,
        "from asr_datamodule import LibriSpeechAsrDataModule",
        "from vi_datamodule import ViAsrDataModule as LibriSpeechAsrDataModule",
    )
    decode = patch(
        decode,
        "    return parser\n",
        '    parser.add_argument(\n'
        '        "--eval-sets",\n'
        '        type=str,\n'
        '        default="valid",\n'
        '        help="Comma-separated cut set names (files in --manifest-dir, without .jsonl.gz).",\n'
        '    )\n'
        "    return parser\n",
    )
    start = decode.index("    test_clean_cuts = librispeech.test_clean_cuts()")
    end = decode.index('    test_dl = [test_clean_dl, test_other_dl]\n') + len(
        '    test_dl = [test_clean_dl, test_other_dl]\n'
    )
    decode = (
        decode[:start]
        + '    test_sets = args.eval_sets.split(",")\n'
        + "    test_dl = [\n"
        + "        librispeech.test_dataloaders(librispeech.eval_cuts(name)) for name in test_sets\n"
        + "    ]\n"
        + decode[end:]
    )
    (args.out / "decode.py").write_text(decode)
    print(f"recipe written to {args.out}")


if __name__ == "__main__":
    main()
