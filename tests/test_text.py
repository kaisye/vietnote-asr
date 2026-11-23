import unicodedata

import pytest

from vnasr.text import normalize_text, strip_tones


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("hoà", "hòa"),
        ("hòa", "hòa"),
        ("khoẻ", "khỏe"),
        ("thuỷ", "thủy"),
        ("HOÀ", "hòa"),
    ],
)
def test_open_syllables_use_one_tone_spelling(variant, canonical):
    assert normalize_text(variant) == canonical


@pytest.mark.parametrize("word", ["hoàn", "toán", "huýt", "khuỷu", "quý", "quỳnh", "thuyết"])
def test_words_with_a_single_spelling_are_unchanged(word):
    assert normalize_text(word) == word


def test_decomposed_input_becomes_nfc():
    assert normalize_text(unicodedata.normalize("NFD", "tiếng Việt")) == "tiếng việt"


def test_punctuation_is_removed_and_apostrophes_inside_words_are_kept():
    assert normalize_text("Don’t, nhé: 'deadline'!  OK?") == "don't nhé deadline ok"


def test_numbers_become_spoken_and_english_is_kept():
    assert normalize_text("Họp lúc 3 giờ với team") == "họp lúc ba giờ với team"


def test_tone_rule_ignores_words_starting_with_gi():
    assert normalize_text("gì giá già") == "gì giá già"


def test_alphanumeric_codes_are_spelled():
    # Changed 2026-11-21 (normalizer 1.3): the BPE has no digit pieces, so a code left as "h5n1" became <unk>.
    assert normalize_text("Virus H5N1 và iPhone 15") == "virus h năm n một và iphone mười lăm"
    assert normalize_text("mô hình B2B") == "mô hình b2b"  # read in English: left for the label filter


def test_strip_tones_keeps_vowel_qualities():
    assert strip_tones("tiếng việt mở đường ơi") == "tiêng viêt mơ đương ơi"


@pytest.mark.parametrize("raw, spoken", [
    ("Chuyển 50.000đ cho anh", "chuyển năm mươi nghìn đồng cho anh"),
    ("Chuyển 1.500.000 đồng", "chuyển một triệu năm trăm nghìn đồng"),
    ("nạp 50k", "nạp năm mươi nghìn"),
    ("lương 15tr", "lương mười lăm triệu"),
    ("Hẹn lúc 7h30 ngày 15/11", "hẹn lúc bảy giờ ba mươi ngày mười lăm tháng mười một"),
    ("lúc 20:30", "lúc hai mươi giờ ba mươi"),
    ("lúc 9:00 sáng", "lúc chín giờ sáng"),
    ("ngày 15/11/2026", "ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu"),
    ("hạn chót 15/11 nhé", "hạn chót mười lăm tháng mười một nhé"),
    ("sáng thứ hai 3/11", "sáng thứ hai ba tháng mười một"),
    ("nhà số 15/11 Lê Duẩn", "nhà số mười lăm trên mười một lê duẩn"),
    ("tỉ lệ 3/4", "tỉ lệ ba trên bốn"),
    ("hẻm 120/45", "hẻm một trăm hai mươi trên bốn mươi lăm"),
    ("iphone 3g và 4g", "iphone ba g và bốn g"),
    ("khoảng 10:00 – 11:00 đêm", "khoảng mười giờ đến mười một giờ đêm"),
    ("vào khoảng 12:00 giờ GMT", "vào khoảng mười hai giờ gmt"),
    ("tỉ số 2-1", "tỉ số hai một"),
])
def test_money_clock_and_dates_are_spoken(raw, spoken):
    # 2026-11-10: "50.000đ" used to become "năm mươi 000đ" and "7h30" stayed as written.
    assert normalize_text(raw) == spoken
