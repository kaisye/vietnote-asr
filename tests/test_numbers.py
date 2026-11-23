import pytest

from vnasr.numbers import spell_integer, spell_number, spell_numbers_in_text


@pytest.mark.parametrize(
    ("value", "spoken"),
    [
        (0, "không"),
        (5, "năm"),
        (10, "mười"),
        (11, "mười một"),
        (14, "mười bốn"),
        (15, "mười lăm"),
        (20, "hai mươi"),
        (21, "hai mươi mốt"),
        (24, "hai mươi tư"),
        (25, "hai mươi lăm"),
        (100, "một trăm"),
        (105, "một trăm linh năm"),
        (110, "một trăm mười"),
        (115, "một trăm mười lăm"),
        (1000, "một nghìn"),
        (1005, "một nghìn không trăm linh năm"),
        (1500, "một nghìn năm trăm"),
        (2026, "hai nghìn không trăm hai mươi sáu"),
        (1_000_000, "một triệu"),
        (1_000_005, "một triệu không trăm linh năm"),
        (2_500_000, "hai triệu năm trăm nghìn"),
        (3_000_000_000, "ba tỷ"),
    ],
)
def test_spell_integer(value, spoken):
    assert spell_integer(value) == spoken


@pytest.mark.parametrize(
    ("token", "spoken"),
    [
        ("1.000", "một nghìn"),
        ("1,000,000", "một triệu"),
        ("3,5", "ba phẩy năm"),
        ("0,5", "không phẩy năm"),
        ("0912", "không chín một hai"),
        ("1.5.2", "một chấm năm chấm hai"),
    ],
)
def test_spell_number_forms(token, spoken):
    assert spell_number(token) == spoken


def test_percent_and_sentence_final_dot():
    assert spell_numbers_in_text("tăng 15% lên 20.") == "tăng mười lăm phần trăm lên hai mươi."


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        # Identifiers are read digit by digit (2026-11-21).
        ("số 086 222 3252", "số không tám sáu hai hai hai ba hai năm hai"),
        ("mã OTP 3252", "mã OTP ba hai năm hai"),
        ("tài khoản 32521234", "tài khoản ba hai năm hai một hai ba bốn"),
        ("stk: 1903 4567", "stk: một chín không ba bốn năm sáu bảy"),
        # Amounts keep their full reading.
        ("dân số 98000000 người", "dân số chín mươi tám triệu người"),
        ("chuyển 3252000 đồng", "chuyển ba triệu hai trăm năm mươi hai nghìn đồng"),
        ("số nhà 3252", "số nhà ba nghìn hai trăm năm mươi hai"),
        # Ordinals (2026-11-21).
        ("ngày 30/4/1975", "ngày ba mươi tháng tư năm một nghìn chín trăm bảy mươi lăm"),
        ("thứ 4 tuần sau", "thứ tư tuần sau"),
        ("lần thứ 1", "lần thứ nhất"),
        ("người thứ 40", "người thứ bốn mươi"),
    ],
)
def test_identifiers_and_ordinals(written, spoken):
    assert spell_numbers_in_text(written) == spoken
