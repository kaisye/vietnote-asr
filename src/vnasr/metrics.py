"""Corpus-level ASR metrics.

Rates are total errors divided by total reference units, not an average of
per-utterance rates, so long utterances keep their proportional weight.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vnasr.text import normalize_text, strip_tones


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """Return a minimum-edit alignment as (ref_item, hyp_item) pairs.

    None marks the missing side of an insertion or a deletion.
    """
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )

    pairs: list[tuple[str | None, str | None]] = []
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


@dataclass(frozen=True)
class Score:
    # Error rate over space-separated units. For Vietnamese these are syllables,
    # so this is the SyER used by VLSP and does not compare with English WER.
    syer: float
    cer: float
    # Among aligned syllables with the same letters, the share with a different
    # tone. Tone errors on syllables whose letters also differ are not visible.
    ter: float
    # Share of listed English reference words recognized; None without a list.
    english_recall: float | None
    ref_syllables: int


def _rate(errors: int, total: int) -> float:
    if total:
        return errors / total
    return 0.0 if errors == 0 else float("inf")


def score(
    pairs: Iterable[tuple[str, str]],
    english_words: set[str] | None = None,
) -> Score:
    """Score (reference, hypothesis) pairs after normalizing both sides."""
    token_errors = char_errors = tone_errors = same_letters = 0
    ref_syllables = ref_chars = english_total = english_correct = 0

    for raw_ref, raw_hyp in pairs:
        ref, hyp = normalize_text(raw_ref), normalize_text(raw_hyp)
        ref_tokens = ref.split()
        ref_syllables += len(ref_tokens)
        ref_chars += len(ref)

        for ref_token, hyp_token in align(ref_tokens, hyp.split()):
            token_errors += ref_token != hyp_token
            if (
                ref_token is not None
                and hyp_token is not None
                and strip_tones(ref_token) == strip_tones(hyp_token)
            ):
                same_letters += 1
                tone_errors += ref_token != hyp_token
            if english_words is not None and ref_token in english_words:
                english_total += 1
                english_correct += ref_token == hyp_token

        char_errors += sum(r != h for r, h in align(ref, hyp))

    english_recall = None
    if english_words is not None and english_total:
        english_recall = english_correct / english_total

    return Score(
        syer=_rate(token_errors, ref_syllables),
        cer=_rate(char_errors, ref_chars),
        ter=_rate(tone_errors, same_letters),
        english_recall=english_recall,
        ref_syllables=ref_syllables,
    )
