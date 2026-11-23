"""User hotword list for contextual biasing plus display forms (2026-11-22).

File format, one name per line ('#' starts a comment):

    Nvidia | en vi đi a
    Jensen Huang
    VinFast | vin phát

The part before '|' is how the name is written. After it come optional Vietnamese readings, separated by ',',
of at least two syllables each: the model writes what it hears ("vin phát"), and a reading is replaced by the
name only as a whole phrase. Both the lowercase name and its readings are boosted in beam search (sherpa-onnx
hotwords), and in the final text the name gets its written casing back.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hotwords:
    names: dict = field(default_factory=dict)  # lowercase name -> written form
    readings: dict = field(default_factory=dict)  # lowercase reading -> written form

    def boost_phrases(self) -> list:
        return sorted({*self.names, *self.readings})

    def apply(self, text: str) -> str:
        pairs = {**self.readings, **self.names}
        for key in sorted(pairs, key=len, reverse=True):
            pat = r"(?<!\w)" + r"\s+".join(map(re.escape, key.split())) + r"(?!\w)"
            text = re.sub(pat, lambda m, v=pairs[key]: v, text, flags=re.IGNORECASE)
        return text


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", " ".join(s.split()))


def load(path: Path) -> Hotwords:
    hw = Hotwords()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, rest = line.partition("|")
        name = _nfc(name)
        if not name:
            continue
        hw.names[name.lower()] = name
        for r in rest.split(","):
            r = _nfc(r).lower()
            if len(r.split()) >= 2:
                hw.readings[r] = name
    return hw


def write_sherpa_files(hw: Hotwords, bpe_model: Path, out_dir: Path) -> tuple:
    """(hotwords_file, bpe_vocab) for sherpa_onnx with modeling_unit='bpe'."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab, words = out_dir / "bpe.vocab", out_dir / "hotwords.sherpa.txt"
    vocab.write_text("".join(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n" for i in range(sp.get_piece_size())),
                     encoding="utf-8")
    words.write_text("".join(p + "\n" for p in hw.boost_phrases()), encoding="utf-8")
    return words, vocab
