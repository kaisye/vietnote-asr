import pytest

from vnasr.metrics import align, score


def test_align_marks_insertions_and_deletions():
    assert align(["a", "b", "c"], ["a", "c"]) == [("a", "a"), ("b", None), ("c", "c")]
    assert align(["a", "c"], ["a", "b", "c"]) == [("a", "a"), (None, "b"), ("c", "c")]


def test_perfect_match_after_normalization():
    result = score([("Hoà bình!", "hòa bình")])
    assert result.syer == 0
    assert result.cer == 0


def test_digits_and_spoken_numbers_compare_equal():
    assert score([("lúc 3 giờ", "lúc ba giờ")]).syer == 0


def test_syer_is_corpus_level_not_utterance_average():
    result = score([("một", "hai"), ("một hai ba", "một hai ba")])
    assert result.syer == pytest.approx(1 / 4)


def test_ter_counts_tone_errors_among_syllables_with_the_same_letters():
    result = score([("tôi mua má", "tôi mua mà")])
    assert result.syer == pytest.approx(1 / 3)
    assert result.ter == pytest.approx(1 / 3)


def test_ter_ignores_syllables_with_different_letters():
    result = score([("tôi mua má cá", "tôi mua mà bò")])
    assert result.ter == pytest.approx(1 / 3)


def test_non_tone_substitution_is_not_a_tone_error():
    assert score([("tôi mua má", "tôi mua cá")]).ter == 0


def test_english_recall_counts_missed_and_deleted_words():
    result = score(
        [("em book meeting với team", "em búc meeting với")],
        english_words={"book", "meeting", "team"},
    )
    assert result.english_recall == pytest.approx(1 / 3)


def test_english_recall_is_none_without_english_words_in_reference():
    assert score([("xin chào", "xin chào")], english_words={"team"}).english_recall is None
    assert score([("xin chào", "xin chào")]).english_recall is None


def test_empty_reference_with_insertions_is_not_reported_as_perfect():
    assert score([("", "xin chào")]).syer == float("inf")
    assert score([("", "")]).syer == 0
