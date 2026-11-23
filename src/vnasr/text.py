"""Text normalization shared by training labels and evaluation."""

import re
import unicodedata

from vnasr.numbers import spell_numbers_in_text

# Bump when output changes: labels and scores from different versions do not compare.
NORMALIZER_VERSION = "1.3"  # 2026-11-21: identifiers digit by digit, codes, padded dates, math

# Vietnamese tone marks in decomposed form. Circumflex, breve and horn are
# vowel qualities, not tones, so they are not listed here.
_TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}

# Open syllables ending in oa, oe, uy have two spellings (hoà/hòa). This project
# uses the form with the tone on the first vowel. A following letter means the
# syllable is closed (hoàn, huýt, khuỷu), where only one spelling exists, and
# "qu" is a single onset, so quý and quỳ are left alone.
_TONE_SHIFT = {}
for _first, _seconds in (("o", "àáảãạ"), ("o", "èéẻẽẹ"), ("u", "ỳýỷỹỵ")):
    for _toned in _seconds:
        _base, _tone = unicodedata.normalize("NFD", _toned)
        _TONE_SHIFT[_first + _toned] = unicodedata.normalize("NFC", _first + _tone) + _base

_TONE_SHIFT_RE = re.compile(r"(?<!q)(" + "|".join(_TONE_SHIFT) + r")(?!\w)")


def _keep_char(text: str, index: int) -> bool:
    char = text[index]
    if unicodedata.category(char)[0] in "LMN":
        return True
    if char != "'":
        return False
    return 0 < index < len(text) - 1 and text[index - 1].isalnum() and text[index + 1].isalnum()


def normalize_text(text: str) -> str:
    """Return lowercase NFC text with one tone spelling, spoken numbers and no punctuation."""
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    text = spell_numbers_in_text(text)
    text = _TONE_SHIFT_RE.sub(lambda match: _TONE_SHIFT[match.group(1)], text)
    kept = "".join(char if _keep_char(text, i) else " " for i, char in enumerate(text))
    return " ".join(kept.split())


def strip_tones(text: str) -> str:
    """Remove tone marks and keep vowel qualities: "tiếng" becomes "tiêng"."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _TONE_MARKS))
