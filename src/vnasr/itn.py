"""Inverse text normalisation: turn spoken Vietnamese numbers in ASR output back into digits.

The recogniser writes numbers the way they are read ("hai nghìn không trăm hai mươi lăm", "hai không hai lăm").
Apps want "2025". The work is done in three steps:

1. Collect spans of number words. Fillers inside a number ("hai nghìn ờ hai mươi hai") and explicit self-corrections
   ("ba à không bốn triệu") are handled here.
2. Segment each span into numbers. A span may hold several numbers ("hai nghìn mười sáu hai nghìn mười tám",
   "mười tám mười chín"), a number and a noun ("hai mươi năm"), a phone number read in groups, or a restart
   ("một nghìn chín trăm bốn bảy mươi lăm" = 1975, the "bốn" abandoned).
3. Decide, from the words around it, whether a short number is a numeral. A lone small number stays a word
   ("một năm đặc biệt", "hai người", "ba mẹ"). It becomes a digit after a context word ("tháng hai", "số một",
   "lớp năm"), before a unit ("hai giờ", "năm phần trăm"), and after a word that is not Vietnamese ("pi bốn",
   "python ba", "task hai"), unless a counting word follows ("react một cái", "team năm người").

Colloquial readings are accepted: "hai lăm" (25), "trăm hai" (120), "hai nghìn mốt" (2100), "ba triệu tư"
(3,4 triệu), "ba triệu rưỡi", "hai chục", "mười một nghìn sáu trăm tỷ" (11.600 tỷ), "mươi nhăm", "ngàn", "lẻ".

Measured with scripts/itn_eval.py (round trip over written text, plus re-readings the way people speak).
History: 2026-11-11 first version; 2026-11-21 rewritten around segmentation after a day of demo bug reports.
"""

import re

_DIGIT = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5,
          "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "linh": 0, "lẻ": 0}
_SCALE = {"nghìn": 1000, "ngàn": 1000, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
_STRUCT = {"mười", "mươi", "trăm", "chục", *_SCALE}
_NUMBER_WORDS = set(_DIGIT) | _STRUCT | {"rưỡi"}
# Words that only continue a number; a span never starts with them.
_NO_START = {"linh", "lẻ", "mươi", "trăm", "rưỡi", "chục", "lăm", "nhăm", "mốt", "tư", *_SCALE}
# Digit words that only appear in shortened readings after tens or a scale ("hai mươi mốt", "triệu tư").
_SHORT_ONLY = {"mốt", "tư", "lăm", "nhăm"}

# A scale word followed by one of these is part of another word ("tỷ lệ", "triệu chứng").
_SCALE_COMPOUND = {"lệ", "số", "trọng", "giá", "suất", "phú", "muội", "chứng", "tập", "hồi", "gọi", "dân"}
# "tư" that begins a span is a number only after these ("tháng tư", "thứ tư", "mùng tư"); otherwise it is
# "thông tư", "đầu tư", "tư vấn".
_TU_CONTEXT = {"tháng", "thứ", "mùng", "mồng", "ngày", "số", "lớp", "hạng", "lần", "phần", "chương", "điều", "khoản"}
# "tư" followed by one of these is a word ("tư vấn", "tư nhân"), except right after "mươi" ("hai mươi tư nhân viên").
_TU_COMPOUND = {"vấn", "nhân", "duy", "liệu", "cách", "thế", "tưởng", "pháp", "lệnh", "bản", "sản", "thục", "trang"}

# Filler sounds that may sit inside a number ("hai nghìn ờ hai mươi").
_FILLERS = {"ờ", "ơ", "ừm", "ừ", "à", "ư", "ừa", "hừm", "ầm"}
# Self-correction markers between two numbers: "ba à không bốn" -> 4. "không phải" is left out on purpose:
# "hai mươi không phải ba mươi" states a contrast, it does not correct.
_CORRECTIONS = (("à", "không"), ("ờ", "không"), ("ơ", "không"), ("à", "nhầm"), ("ờ", "nhầm"), ("nhầm",),
                ("à", "lộn"), ("ờ", "lộn"), ("ý", "lộn"), ("lộn",), ("à", "quên"), ("ý", "nhầm"))

# Words that make a following small number a numeral ("tháng hai" -> "tháng 2").
_BEFORE = {"tháng", "ngày", "mùng", "mồng", "lúc", "số", "phòng", "lớp", "khu", "quận", "phường", "hẻm", "ngõ",
           "kiệt", "đường", "trang", "chương", "câu", "bài", "điểm", "top", "level", "version", "bản", "thứ", "phẩy",
           "điều", "khoản", "mục", "vòng", "hạng", "tập", "kỳ", "kì", "khóa", "khoá", "tầng", "lầu", "dãy", "lô",
           "tuyến", "cấp", "bảng", "hình", "mã", "otp", "pin", "code", "hotline", "zalo", "cccd", "cmnd", "sđt", "đt",
           "stk", "covid", "vít", "phiên", "cổng", "chuyến", "hiệp", "mùa", "đợt"}
# Context words strong enough to win over a counting word after the number ("tháng hai năm", "số một người").
_STRONG_BEFORE = {"tháng", "ngày", "mùng", "mồng", "thứ", "phẩy", "số", "mã", "otp", "pin", "code"}
# Two-word compounds whose second word is a context word on its own ("cung cấp một", "bất kỳ một").
_NOT_BEFORE2 = {"cung cấp", "phân cấp", "trợ cấp", "khẩn cấp", "nâng cấp", "cấp cấp", "bất kỳ", "bất kì", "định kỳ",
                "chu kỳ", "thời kỳ", "cùng lúc", "đúng lúc", "có lúc", "nhiều lúc", "đôi lúc", "mọi lúc", "phần mềm",
                "tuyến đường", "con đường", "trên đường", "ra đường", "dọc đường", "thí điểm", "đặc điểm", "địa điểm",
                "quan điểm", "thời điểm", "ưu điểm", "nhược điểm", "điểm điểm"}
# Two-word context words, matched on the two previous words ("nghị định bảy", "giai đoạn hai").
_BEFORE2 = {"nghị định", "nghị quyết", "thông tư", "chỉ thị", "quyết định", "công văn", "công điện", "giai đoạn",
            "học kỳ", "học kì", "tài khoản", "điện thoại", "số nhà", "quốc lộ", "tỉnh lộ", "cao tốc", "ví dụ"}
# Units after a number that make it a numeral ("hai giờ" -> "2 giờ").
# Durations ("một năm", "hai tháng") are left out on purpose: "một năm đặc biệt" must stay words.
_AFTER = {"giờ", "phút", "giây", "tuổi", "kg", "g", "km", "m", "mét", "ki", "cm", "mm", "lít", "ml", "độ", "phẩy",
          "đồng", "usd", "đô", "k", "gb", "mb", "tb", "inch", "%", "ghz", "mp", "fps", "hz", "kw", "mah", "gam"}
# A unit word followed by one of these is part of another word ("một đồng chí", "một tuổi thơ", "độ tuổi").
_UNIT_COMPOUND = {
    "đồng": {"chí", "thời", "nghiệp", "ý", "hồ", "đội", "bào", "minh", "hành", "loại", "nhất", "tình", "cảm", "ruộng",
             "bằng", "phục", "dạng", "môn", "hương", "tâm", "thuận", "lòng", "sáng", "tác", "vọng", "lúa", "nát",
             "dao", "bộ", "khởi", "nai", "thanh", "đều", "ca", "nghĩa", "phạm", "quê"},
    "tuổi": {"thơ", "trẻ", "tác", "teen", "già", "xế", "đời", "học", "thọ", "tên"},
    "độ": {"tuổi", "dài", "ẩm", "cao", "rộng", "sâu", "nhạy", "chính", "bền", "phân", "tin", "lệch", "khó", "mặn",
           "sáng", "trễ", "phủ", "dày", "lớn", "nhớt", "cứng", "tương", "đặc", "an"},
    "phút": {"chốc", "giây", "cuối"}, "giây": {"lát", "phút"}, "giờ": {"đây", "giấc"},
}
# Counting words: a number before one of these is a quantity and stays a word ("một cái", "năm người").
# Vietnamese classifiers are a closed class, unlike the open set of product names they replace (2026-11-21).
_COUNTERS = {"cái", "con", "chiếc", "người", "lần", "lượt", "tiếng", "ngày", "đêm", "tuần", "tháng", "năm", "buổi",
             "bữa", "cuộc", "cuốn", "quyển", "tờ", "bài", "câu", "chữ", "đứa", "bạn", "anh", "chị", "em", "ông", "bà",
             "cô", "chú", "thằng", "vị", "bên", "phía", "cách", "kiểu", "loại", "dạng", "bước", "phần", "chút", "tí",
             "ít", "cặp", "đôi", "bộ", "ly", "cốc", "chai", "lon", "gói", "hộp", "trái", "quả", "cây", "căn", "mặt",
             "khúc", "nhóm", "ván", "trận", "hiệp", "ca", "lớp", "môn", "điều", "việc", "thứ", "chỗ", "nơi", "cục",
             "miếng", "viên", "tấm", "đợt", "kỳ", "mùa", "đời", "mũi", "liều", "tiết", "khâu", "vòng", "đoạn",
             "dòng", "hàng", "cột", "ô", "khối", "phòng", "tầng", "bản", "trang", "điểm", "ý", "vấn"}
# Common English words after which a number is still a quantity ("ok hai người", "like ba cái").
_EN_FUNCTION = {"ok", "okay", "yes", "no", "hi", "bye", "hello", "oh", "wow", "sorry", "thanks", "the", "an", "of",
                "in", "on", "at", "for", "and", "or", "but", "so", "like", "very", "really", "yeah", "yep", "uh", "um",
                "hmm", "please", "right", "well", "then", "now", "just", "with", "about", "only", "maybe", "is", "are",
                "be", "it", "this", "that", "we", "you", "they", "my", "your", "our", "by", "from", "as", "if", "not"}
_COUNTERS |= {"sao", "thứ", "series"}
# Words after which "một" is the article "a" ("một số", "một vài", "một lúc").
_MOT_ARTICLE = {"số", "vài", "chút", "ít", "lúc", "khi", "trong", "mình", "thời", "hồi", "tí", "tẹo", "xíu", "cách",
                "kiểu", "loạt", "nửa", "mực", "khác", "lát", "phát", "thôi", "mạch", "hơi", "chiều", "lèo", "mai"}
# "thứ hai" (Monday) and "thứ ba" keep their word form.
_KEEP_AFTER_THU = {"hai", "ba", "tư", "năm", "sáu", "bảy"}
# Words before a time noun that make the next number a rate ("mỗi ngày một lần", "một tháng hai lần").
_RATE = {"mỗi", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "vài", "mấy"}
_TIME_NOUN = {"ngày", "tháng", "tuần", "năm", "buổi"}
# Southern names by birth order: "ông Mười", "anh Tư", "chị Ba", "cô Sáu" are people, not numbers.
_KIN = {"ông", "bà", "anh", "chị", "cô", "chú", "bác", "dì", "cậu", "thím", "mợ", "thằng", "con", "út", "má", "tía",
        "dượng", "o", "ả", "cụ"}
# Spoken separators between two numbers: "hẻm mười hai xẹt ba" -> 12/3, "hai ba gạch ngang bốn" -> 23-4.
_SLASH_WORDS = ("xẹt", "xoẹt", "xuyệt", "xuộc", "chéo")
_DASH_WORDS = ("ngang", "nối")
_RATE_UNITS = {"lần", "buổi", "bữa", "lượt", "đợt", "cữ", "liều", "viên", "ca", "cốc", "ly", "gói"}
# After these "năm" is the noun year ("năm nay", "năm trước").
_YEAR_NOUN_NEXT = {"nay", "ngoái", "sau", "trước", "tới", "qua", "rồi", "đó", "ấy", "học", "tài", "mới", "xưa", "kia",
                   "đầu", "cuối", "nào", "này", "kế", "ấy"}
# IDs after which a run of "không" is zeros ("mã không không một").
_ID_PREV = {"số", "mã", "otp", "pin", "code", "phẩy", "stk", "khoản", "thoại", "hotline", "zalo", "sđt"}

_VI_CHARS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
_VI_ASCII_SYLLABLE = re.compile(
    r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|r|s|t|v|x)?"
    r"(?:oai|oay|oao|oeo|uya|uyu|uay|ai|ao|au|ay|eo|ia|iu|oa|oe|oi|ua|ui|uy|a|e|i|o|u|y)"
    r"(?:ch|ng|nh|c|m|n|p|t)?")


def is_foreign(word: str) -> bool:
    """True for a word that cannot be a Vietnamese syllable ("pi", "iphone", "task", "rtx"), by spelling alone."""
    w = word.lower()
    if not w.isalpha() or not w.isascii() or any(c in _VI_CHARS for c in w):
        return False
    return _VI_ASCII_SYLLABLE.fullmatch(w) is None


# ----------------------------------------------------------------------------------------------------------- parse
def _parse_below_thousand(words, short=True, head=False):
    """Parse a reading below 1000 with optional trăm/mươi/mười/linh/chục; None if it is not one."""
    if not words:
        return None
    value, i, n = 0, 0, len(words)
    if i + 1 < n and words[i] in _DIGIT and words[i] not in _SHORT_ONLY and words[i + 1] == "trăm":
        value += _DIGIT[words[i]] * 100
        i += 2
        if i < n and words[i] in ("linh", "lẻ"):
            if i + 2 == n and words[i + 1] in _DIGIT and words[i + 1] not in _SHORT_ONLY:
                return value + _DIGIT[words[i + 1]]
            return None
    elif words[0] == "trăm" and short and n > 1:
        value, i = 100, 1  # "trăm hai" = 120, "trăm rưỡi" = 150: the "một" dropped
    if i == n:
        return value
    rest = words[i:]
    if i and len(rest) == 1 and short:
        # Shortened "một trăm hai" = 120, "trăm rưỡi" = 150. Not "năm" ("một trăm năm" is a hundred years), not
        # "một"/"không" (too often the next word).
        if rest[0] == "rưỡi":
            return value + 50
        if rest[0] in _DIGIT and rest[0] not in ("không", "một", "năm", "linh", "lẻ"):
            return value + _DIGIT[rest[0]] * 10
    # Colloquial "một trăm bốn lăm" = 145 (mươi dropped after trăm).
    if i and len(rest) == 2 and (short or rest[1] in _SHORT_ONLY) and all(w in _DIGIT for w in rest) and rest[0] not in ("không", "linh", "lẻ", "lăm", "mốt") \
            and rest[1] not in ("linh", "lẻ", "không"):
        return value + _DIGIT[rest[0]] * 10 + _DIGIT[rest[1]]
    # Only in the group before a scale word ("sáu tư nghìn" = 64.000); after one, "ba nghìn tám một ký" is 3.800 + "một ký".
    if head and not i and len(rest) == 2 and _is_two_digit(rest) or head and (not i and len(rest) == 2 and rest[1] in _SHORT_ONLY
                                                          and rest[0] in _DIGIT and _DIGIT[rest[0]] >= 2):
        if short:
            return _DIGIT[rest[0]] * 10 + _DIGIT[rest[1]]  # "sáu tư nghìn" = 64.000, "hai tám" = 28
    if len(rest) == 2 and rest[0] in _DIGIT and rest[0] not in _SHORT_ONLY and rest[1] == "chục" and _DIGIT[rest[0]]:
        return value + _DIGIT[rest[0]] * 10  # "hai chục" = 20
    if rest[0] == "mười":
        tens, j = 10, 1
    elif len(rest) >= 2 and rest[0] in _DIGIT and rest[1] == "mươi" and _DIGIT[rest[0]] >= 2 and rest[0] not in _SHORT_ONLY:
        tens, j = _DIGIT[rest[0]] * 10, 2
    elif len(rest) == 1 and rest[0] in _DIGIT and rest[0] not in ("linh", "lẻ", *_SHORT_ONLY):
        # "một trăm năm" is 100 years (105 is "một trăm linh năm"); a bare digit after trăm is not part of the number.
        return None if i else _DIGIT[rest[0]]
    else:
        return None
    if j == len(rest):
        return value + tens
    # After mười/mươi five is "lăm" and one is "mốt"; "hai mươi năm" is twenty years, "mười năm" ten years.
    if j + 1 == len(rest) and rest[j] in _DIGIT and rest[j] not in ("linh", "lẻ", "năm", "không"):
        if rest[j] == "mốt" and tens == 10:
            return None
        return value + tens + _DIGIT[rest[j]]
    return None


_SCALE_ORDER = (("tỷ", "tỉ"), ("triệu",), ("nghìn", "ngàn"))


def _parse_scaled(words, level, short):
    for li in range(level, len(_SCALE_ORDER)):
        names = _SCALE_ORDER[li]
        k = next((idx for idx, w in enumerate(words) if w in names), None)
        if k is None:
            continue
        if k == 0:
            return None
        scale = _SCALE[names[0]]
        left_words = words[:k]
        # "mười một nghìn sáu trăm tỷ" = 11.600 tỷ: a count of thousands may stand before tỷ.
        if li == 0 and any(w in ("nghìn", "ngàn") for w in left_words):
            left = _parse_scaled(left_words, 2, short)
        else:
            left = _parse_below_thousand(left_words, short, head=True)
        if left is None or (left == 0 and level == 0 and li == 0):
            return None
        right = words[k + 1:]
        if not right:
            return left * scale
        if len(right) == 1 and right[0] not in _STRUCT:
            if not short:
                return None
            if right[0] == "rưỡi":  # "ba triệu rưỡi", "hai nghìn rưỡi"
                return left * scale + scale // 2
            # Shortened "ba triệu tư" = 3,4 triệu, "hai nghìn mốt" = 2100. After nghìn only the unambiguous forms:
            # "hai nghìn hai" could be 2002 or 2200. "ba triệu năm" is more likely "three million years".
            # "ba nghìn tám" = 3.800 (prices); years use short=False, so "hai nghìn hai" is never 2200 there.
            # Not "một": "hai triệu một tháng" is per month (the shortened one is "mốt").
            if right[0] in _DIGIT and right[0] not in ("không", "một", "năm", "linh", "lẻ"):
                return left * scale + _DIGIT[right[0]] * scale // 10
            return None
        rest = _parse_scaled(right, li + 1, short)
        if rest is None:
            return None
        return left * scale + rest
    return _parse_below_thousand(words, short)


def parse_number(words, short=True):
    """Parse a full spoken number ("hai nghìn không trăm hai mươi lăm") into an int, or None.

    `short=False` refuses shortened readings ("trăm hai", "nghìn mốt"); years use it.
    """
    words = list(words)
    if not words or not any(w in _STRUCT for w in words):
        return None
    return _parse_scaled(words, 0, short)


def _digit_string(words):
    if words and all(w in _DIGIT and w not in ("linh", "lẻ") for w in words):
        # "tư", "lăm", "mốt", "nhăm" only follow a tens digit: "hai lăm" = 25, but never "lăm hai" or "ba nhăm một".
        if len(words) > 1 and any(w in _SHORT_ONLY and (k == 0 or k != len(words) - 1) for k, w in enumerate(words)):
            return None
        return "".join(str(_DIGIT[w]) for w in words)
    return None


def _format_value(value):
    """Write a full reading the way Vietnamese text does: millions and billions as "3,4 triệu" / "1,25 tỷ".

    A value that needs more than three decimals there ("3.400.500") is written with dot thousand separators, and so
    is a count of billions of a thousand or more ("11.600 tỷ").
    """
    for scale, word in ((10**9, "tỷ"), (10**6, "triệu")):
        if value >= scale:
            whole, rest = divmod(value, scale)
            if rest % (scale // 1000):
                return [f"{value:,}".replace(",", ".")]
            frac = f"{rest // (scale // 1000):03d}".rstrip("0")
            head = f"{whole:,}".replace(",", ".")
            return [f"{head},{frac}" if frac else head, word]
    # Below a million: dot thousand separators ("250.000", "38.000", "2.500"). Values 1000–2100 stay plain because
    # they are usually years ("năm 1537", "2025"), which Vietnamese text never writes with a dot.
    if value >= 10000 or 2100 < value < 10000:
        return [f"{value:,}".replace(",", ".")]
    return [str(value)]


# ----------------------------------------------------------------------------------------------------- segmentation
def _colloquial_year(words):
    """Years read the way people say them, or None.

    "hai nghìn hai hai" / "hai ngàn hai mốt" = 2022 / 2021 (last two digits one by one), "hai không năm mươi" = 2050
    (first two digits one by one), "hai nghìn không trăm hai hai" = 2022.
    """
    if len(words) == 4 and words[1] in ("nghìn", "ngàn") and words[0] == "hai" and words[2] != "không":
        tail = _digit_string(words[2:])
        if tail and len(tail) == 2:
            return _DIGIT[words[0]] * 1000 + int(tail)
    if len(words) == 6 and words[1] in ("nghìn", "ngàn") and words[2:4] == ["không", "trăm"] and words[0] in ("một", "hai"):
        tail = _digit_string(words[4:])
        if tail and len(tail) == 2 and tail[0] != "0":
            return _DIGIT[words[0]] * 1000 + int(tail)
    if len(words) >= 3 and words[:2] in (["hai", "không"], ["một", "chín"]):
        rest = parse_number(words[2:], short=False)
        if rest is not None and 10 <= rest <= 99:
            return (2000 if words[0] == "hai" else 1900) + rest
    return None


def _year(words):
    """A 4-digit year read in full or digit by digit, as a string, else None."""
    value = parse_number(words, short=False)
    if value is None and len(words) >= 4 and words[-3] == "trăm" and all(w in _DIGIT for w in words[-2:]):
        value = parse_number(words)  # "một nghìn ba trăm ba ba" = 1333
    if value is None:
        value = _colloquial_year(words)
    if value is None:
        digits = _digit_string(words)
        value = int(digits) if digits and len(digits) == 4 and digits[0] != "0" else None
    return str(value) if value is not None and 1000 <= value <= 2999 else None


def _phone_groups(words):
    """Phone numbers read in groups: "không tám mươi sáu hai trăm hai mươi hai ba trăm năm mươi hai" -> "086222352".

    Each group is the longest reading below 1000 from the left (or one digit). Accepted only when it gives 9 to 11
    digits starting with 0, the shape of a Vietnamese phone number, so ordinary number runs are never touched.
    """
    if not words or words[0] != "không" or any(w in _SCALE or w == "rưỡi" for w in words):
        return None
    digits, i = "", 0
    while i < len(words):
        for j in range(min(len(words), i + 7), i, -1):
            group = words[i:j]
            value = _parse_below_thousand(group, short=False) if any(w in _STRUCT for w in group) else None
            if value is None and len(group) == 1 and group[0] in _DIGIT and group[0] not in ("linh", "lẻ"):
                value = _DIGIT[group[0]]
            if value is not None:
                digits += str(value)
                i = j
                break
        else:
            return None
    return digits if 9 <= len(digits) <= 11 else None


def _segment_kind(words, short=True):
    """("full", value) for a reading with structure words, ("digits", str) for digit by digit, else None."""
    if any(w in _STRUCT for w in words):
        value = parse_number(words, short)
        if value is None:
            value = _colloquial_year(words)
        return ("full", value) if value is not None else None
    if len(words) == 2 and words[1] in _SHORT_ONLY and words[0] in _DIGIT and _DIGIT[words[0]] >= 2 \
            and words[0] not in _SHORT_ONLY:
        return ("full", _DIGIT[words[0]] * 10 + _DIGIT[words[1]])  # "hai mốt" = 21, "bốn lăm" = 45
    digits = _digit_string(words)
    return ("digits", digits) if digits is not None else None


def _segment(words):
    """Split a span into [(words, kind, value)], kind "full" | "digits" | "word"; None when no clean split exists.

    The best split has the fewest pieces. A lone digit word may only stand at either end ("thứ hai | mười tám",
    "hai mươi | năm"); one in the middle means the speaker restarted, which `_repair` handles. Pieces must be
    standard readings: "chín trăm bốn | bảy mươi lăm" is a restart of 1975, not 940 and 75.
    """
    # A digit-by-digit piece next to a full reading is a restart, not two numbers ("một trăm | hai ba"), so each
    # split uses one kind of number only.
    n = len(words)
    best = [None] * (n + 1)  # best[i] = (cost, pieces) for words[:i]
    best[0] = (0.0, [])
    for end in range(1, n + 1):
        for start in range(end - 1, -1, -1):
            if best[start] is None:
                continue
            piece = words[start:end]
            kind = _segment_kind(piece, short=False)
            if kind is None or (kind[0] == "digits" and len(piece) == 1):
                # A lone word at either end ("thứ hai | mười tám"), or the noun "năm" right after a full number in
                # the middle ("một trăm hai mươi | năm | một trăm hai mươi năm").
                after_full = best[start][1] and best[start][1][-1][1] == "full"
                if len(piece) == 1 and piece[0] in _DIGIT and piece[0] not in _SHORT_ONLY and (
                        start == 0 or end == n or (piece[0] == "năm" and after_full)):
                    kind, cost = ("word", None), 1.1
                else:
                    continue
            else:
                cost = 1.0 if kind[0] == "full" else 1.2 + 0.05 * len(piece)
            total = best[start][0] + cost
            if best[end] is None or total < best[end][0] - 1e-9:
                best[end] = (total, best[start][1] + [(piece, *kind)])
    return best[n][1] if best[n] else None


def _repair(words):
    """A restart inside a number: drop the abandoned words so the rest reads as one number.

    "một nghìn chín trăm bốn bảy mươi lăm" -> 1975 (the speaker said "bốn", then corrected to "bảy mươi lăm").
    The earliest, shortest deletion wins: a correction comes after the mistake.
    """
    for k in range(1, len(words) - 1):
        # The abandoned word is a digit right after trăm/mươi/a scale, and a digit takes its place.
        if words[k] not in _DIGIT or words[k - 1] not in _STRUCT or words[k + 1] not in _DIGIT \
                or words[k + 1] in _SHORT_ONLY:
            continue
        rest = words[:k] + words[k + 1:]
        value = parse_number(rest, short=False)
        if value is not None:
            return rest, value
    return None


def _split_on_year(words):
    """Where to end a span before "năm" + a year ("tháng mười hai năm hai nghìn không trăm mười ba"), or None."""
    if len(words) > 6 and _digit_string(words):
        return None  # a long digit-by-digit run is a phone number or code ("không năm một bốn tám ...")
    for k in range(1, len(words) - 1):
        if words[k] == "năm":
            for end in range(len(words), k + 1, -1):
                if _year(words[k + 1:end]):
                    return k
    return None


def _split_short_nam(words, prev):
    """Short digit runs with "năm" in them: "tháng năm năm nay", "hai ba năm", "ba năm bốn năm" read "năm" as year.

    Four-digit codes such as "ba hai năm hai" keep their "năm".
    """
    if len(words) < 2 or len(words) > 4 or "năm" not in words[1:] or any(w in _STRUCT for w in words):
        return None
    if prev in ("tháng", "ngày", "mùng", "mồng") and words[1] == "năm":
        return 1
    if words[-1] == "năm" and prev not in _ID_PREV:
        return len(words) - 1
    return None


# ------------------------------------------------------------------------------------------------------- context
def _is_counting(digits):
    steps = {int(b) - int(a) for a, b in zip(digits, digits[1:])}
    return steps in ({1}, {-1})


def _is_two_digit(words):
    """ "hai tám" = 28, "sáu chín" = 69: two digits read without "mươi", not neighbours (a range) nor repeats."""
    # Not after "năm": "năm ba cái" (a few), "năm bảy lần" (several times), "năm một" (year one).
    if len(words) != 2 or words[0] in ("không", "một", "mốt", "linh", "lẻ", "năm") or words[1] in ("không", "linh", "lẻ"):
        return False
    a, b = _DIGIT.get(words[0]), _DIGIT.get(words[1])
    return a is not None and b is not None and abs(a - b) > 1



def _is_context_numeral(words, prev, prev2, nxt, next2):
    """Whether a lone small number or a two-digit run ("hai ba") is a numeral here."""
    if len(words) == 1 and prev in _KIN and nxt not in _AFTER:
        return False  # "anh Tư", "chị Ba": a name ("anh hai tuổi" still has its unit)
    if words == ["không"] and prev != "phẩy":
        return False  # "không" alone is "not" ("chuyển đổi số không chỉ")
    if all(w == "không" for w in words) and prev not in _ID_PREV:
        return False  # "không không không" is "no no no"
    if words == ["năm"] and nxt in _YEAR_NOUN_NEXT:
        return False  # "cùng kỳ năm trước", "quyết định năm nay"
    if prev in _TIME_NOUN and prev2 in _RATE and (nxt in _RATE_UNITS or prev2 == "mỗi"):
        return False  # a rate: "mỗi ngày một lần", "hai tháng một lần"
    if prev in ("đến", "tới", "và", "hoặc", "hay", "với", *_SLASH_WORDS, *_DASH_WORDS) and prev2[:1].isdigit():
        return True  # the second end of a range: "gleason 2 đến bốn", "từ 2 đến chín"
    if prev == "ngày" and words == ["một"] and nxt not in ("tháng", ""):
        return False  # "ngày một phát triển" (more and more)
    if prev == "tháng" and prev2 == "ngày" and words == ["năm"]:
        return False  # "ngày tháng năm sinh"
    if prev in _STRONG_BEFORE or (prev == "giờ" and prev2.isdigit()):  # "14 giờ tám" (minutes)
        return True
    # "một" before these is the article "a", not the number one ("một số", "một vài", "một lúc").
    if words == ["một"] and nxt in _MOT_ARTICLE:
        return False
    counted = nxt in _COUNTERS and nxt != prev  # "lớp một lớp hai" is a list, not a count
    if (f"{prev2} {prev}" in _BEFORE2 or prev in _BEFORE) and not counted and f"{prev2} {prev}" not in _NOT_BEFORE2:
        return True
    if nxt in _AFTER and next2 not in _UNIT_COMPOUND.get(nxt, ()):
        return True
    if f"{nxt} {next2}" in _UNIT_STARTS or nxt in ("oát", "vôn"):
        return True  # "tám ghi ga bai", "hai mê ga oát"
    if nxt == "phần" and next2 == "trăm":
        return True
    # After a word that is not Vietnamese ("pi bốn", "python ba", "task hai"), unless a quantity follows.
    # "một" after an English word is mostly the article ("workshop một series", "hemoglobin một protein").
    if is_foreign(prev) and prev not in _EN_FUNCTION and not counted:
        if words not in (["một"], ["năm"]) or nxt in ("", prev):  # "fake một fake hai" is a list
            return True
    return False


# --------------------------------------------------------------------------------------------------------- spans
_FILL, _CORR = "\x00fill", "\x00corr"


def _collect(tokens, i, fix_corrections):
    """Collect the span from tokens[i]: number words, with fillers and correction markers between them.

    Returns (end index, marked) where marked holds lowercase number words and (_FILL, token) / (_CORR,) marks.
    """
    marked, j = [], i
    while j < len(tokens):
        w = tokens[j].lower()
        after = tokens[j + 1].lower() if j + 1 < len(tokens) else ""
        if w in _NUMBER_WORDS:
            if w in _SCALE and after in _SCALE_COMPOUND:
                break  # "tỷ lệ", "triệu chứng"
            if w == "tư" and after in _TU_COMPOUND and not (marked and marked[-1] == "mươi"):
                break  # "thông tư", "đầu tư vấn"
            if marked and w == "năm" and after in _YEAR_NOUN_NEXT:
                break  # "tháng ba năm nay": the year noun, not a digit
            marked.append(w)
            j += 1
            continue
        if marked and fix_corrections:
            hit = next((m for m in _CORRECTIONS if tuple(t.lower() for t in tokens[j:j + len(m)]) == m), None)
            if hit and j + len(hit) < len(tokens):
                follow = tokens[j + len(hit)].lower()
                if follow in _NUMBER_WORDS and follow not in _NO_START:
                    marked.append((_CORR,))
                    j += len(hit)
                    continue
        if marked and w in _FILLERS:
            k = j
            while k < len(tokens) and tokens[k].lower() in _FILLERS:
                k += 1
            if k < len(tokens) and tokens[k].lower() in _NUMBER_WORDS:
                marked.append((_FILL, " ".join(tokens[j:k])))
                j = k
                continue
        break
    # A span never ends on a mark (the number after it was cut off, as in "ờ năm nay").
    while marked and isinstance(marked[-1], tuple):
        marked.pop()
        j -= 1
        while j > i and tokens[j - 1].lower() in _FILLERS:
            j -= 1
    return j, marked


def _groups(marked):
    """Number groups of a marked span after corrections and fillers: (groups, fillers between them)."""
    if (_CORR,) in marked:
        # The correction replaces what came before it: "ba à không bốn triệu" -> "bốn triệu".
        last = len(marked) - 1 - marked[::-1].index((_CORR,))
        marked = marked[last + 1:]
    groups, fillers, current = [], [], []
    for w in marked:
        if isinstance(w, tuple):
            if not current:  # two fillers in a row ("ờ ờ"): keep both between the same groups
                fillers[-1] += " " + w[1]
                continue
            groups.append(current)
            fillers.append(w[1])
            current = []
        else:
            current.append(w)
    groups.append(current)
    # Join the groups around a filler when they read as one number together: "hai nghìn | ờ | hai mươi hai".
    out, out_fillers = [groups[0]], []
    for filler, group in zip(fillers, groups[1:]):
        # The number is unfinished before the filler ("hai nghìn | ờ") or the part after it cannot stand alone
        # ("ờ | không trăm hai mươi", "ờ | linh năm"); segmentation then splits the joined span if it holds more.
        unfinished = out[-1][-1] in _SCALE or out[-1][-1] in ("trăm", "mươi", "mười")
        continues = group[0] in _NO_START or group[:2] == ["không", "trăm"]
        if unfinished or continues:
            out[-1] = out[-1] + group
        else:
            out.append(group)
            out_fillers.append(filler)
    return out, out_fillers


def _range(pieces, original):
    """ "tám chín triệu" -> "8-9 triệu", "ba lăm bốn mươi nghìn" -> "35-40 nghìn": a range before one scale word."""
    tail = []
    if len(pieces) == 3 and pieces[2][1] == "word":
        tail, pieces = original[-1:], pieces[:2]  # "ba lăm bốn mươi nghìn | một cân"
        original = original[:-1]
    if len(pieces) != 2:
        return None
    (w1, k1, v1), (w2, k2, v2) = pieces
    if k2 != "full" or w2[-1] not in _SCALE or v2 is None:
        return None
    scale = _SCALE[w2[-1]]
    lead, rest = divmod(v2, scale)
    if rest or lead >= 1000 or any(w in _SCALE for w in w2[:-1]):
        return None
    if k1 == "word":
        low = _DIGIT[w1[0]]
    elif k1 == "full" and v1 is not None and not any(w in _SCALE for w in w1):
        low = v1
    else:
        return None
    if not low < lead <= low + max(3, low // 2):
        return None
    return [f"{low}-{lead}", original[-1], *tail]


def _render_pieces(words, pieces, prev, prev2, nxt, next2, original):
    ranged = _range(pieces, original)
    if ranged:
        return ranged
    out, pos = [], 0
    for idx, (piece, kind_name, _value) in enumerate(pieces):
        has_next = idx + 1 < len(pieces)
        p_prev = out[-1].lower() if out else prev
        p_prev2 = (out[-2].lower() if len(out) > 1 else prev) if out else prev2
        p_next = pieces[idx + 1][0][0] if has_next else nxt
        p_next2 = (pieces[idx + 1][0][1:2] or [""])[0] if has_next else next2
        span = original[pos:pos + len(piece)]
        pos += len(piece)
        if kind_name == "word":
            if has_next and pieces[idx + 1][0][0] == piece[0] and pieces[idx + 1][1] == "full":
                continue  # a stutter: "hai hai mươi" -> 20
            if idx == 0 and has_next and piece[0] != "không":
                # "tám chín triệu", "bảy tám ngàn", "năm sáu trăm ngàn": a range written in words; "tám 9 triệu"
                # would be wrong, so the whole span stays as spoken.
                return list(original)
            if idx > 0 and pieces[idx - 1][1] == "full" or piece[0] == "không":
                out += span  # the noun after a number ("hai mươi | năm"), the question "không" before one
                continue
        out += _render(piece, p_prev, p_prev2, p_next, p_next2, span)
    return out


def _render(words, prev, prev2, nxt, next2, original):
    """Output tokens for one span of lowercase number words. `original` holds the words as written."""
    if not words:
        return []
    # A trailing "không" is the question particle ("năm hai nghìn không trăm hai mươi mốt không?") when the rest
    # reads as a number without it.
    if len(words) > 2 and words[-1] == "không" and _segment_kind(words) is None:
        core = words[1:-1] if words[0] == "năm" else words[:-1]
        if parse_number(core) is not None:
            return _render(words[:-1], prev, prev2, "không", nxt, original[:-1]) + [original[-1]]
    # Year: "năm" + a 4-digit reading ("năm hai không hai lăm", "năm hai nghìn không trăm hai mươi lăm").
    if words[0] == "năm" and len(words) > 1:
        rest = words[1:]
        if _year(rest) is None and _segment(rest) is None and _phone_groups(rest) is None:
            repaired = _repair(rest)
            if repaired and 1000 <= repaired[1] <= 2999:
                return [original[0], str(repaired[1])]  # "năm một nghìn chín trăm bốn bảy mươi lăm" -> năm 1975
        for end in range(len(words), 1, -1):
            year = _year(words[1:end])
            # Not when the rest continues the number ("... bốn | mươi lăm"): that is a restart, repaired below.
            if year and (end == len(words) or words[end] not in _NO_START):
                return [original[0], year] + _render(words[end:], year, "năm", nxt, next2, original[end:])
    # "đến năm hai mươi tư" is the year 24 said short: the noun "năm" and a full number, unless a count follows
    # ("năm hai mươi người" stays as spoken).
    if words[0] == "năm" and len(words) > 2 and nxt not in _COUNTERS and words[1] not in _NO_START \
            and _segment_kind(words) is None:
        rest = _segment_kind(words[1:])
        if rest and rest[0] == "full":
            return [original[0], *_format_value(rest[1])]
    k = _split_on_year(words)
    if k is not None:
        head = _render(words[:k], prev, prev2, words[k], words[k + 1], original[:k])
        return head + _render(words[k:], head[-1].lower() if head else prev, prev, nxt, next2, original[k:])
    # "thứ hai", "thứ ba" are weekdays.
    if prev == "thứ" and words[0] in _KEEP_AFTER_THU and (len(words) == 1 or _segment_kind(words, False) is None
                                                          or _segment_kind(words)[0] == "digits"):
        return [original[0]] + _render(words[1:], words[0], prev, nxt, next2, original[1:])
    k = _split_short_nam(words, prev)
    if k is not None:
        after = words[k + 1] if k + 1 < len(words) else nxt
        head = _render(words[:k], prev, prev2, "năm", after, original[:k])
        return head + [original[k]] + _render(words[k + 1:], "năm", words[k - 1], nxt, next2, original[k + 1:])

    kind = _segment_kind(words)
    if kind is None:
        phone = _phone_groups(words)
        if phone:
            return [phone]
        # A shortened reading closed by a lone word: "ba nghìn tám | một ký" = 3.800 + "một ký".
        if len(words) >= 3 and words[-1] in _DIGIT and parse_number(words[:-1], short=False) is None:
            head = parse_number(words[:-1])
            if head is not None and words[-2] not in _STRUCT:
                return _format_value(head) + [original[-1]]
        pieces = _segment(words)
        if pieces is not None and len(pieces) > 1:
            return _render_pieces(words, pieces, prev, prev2, nxt, next2, original)
        repaired = _repair(words)
        if repaired:
            return _format_value(repaired[1])
        if len(words) > 2 and words[-1] == "năm":  # "một nghìn một năm": the year noun closes the span
            return _render(words[:-1], prev, prev2, "năm", nxt, original[:-1]) + [original[-1]]
        return list(original)
    if kind[0] == "full":
        if words == ["mười"] and prev in _KIN:
            return list(original)  # "ông Mười"
        return _format_value(kind[1])
    # Digit by digit. One word, or two that may be a range ("hai ba người"), needs context to be a numeral.
    # "hai lăm" / "ba mốt" can only be 25 / 31, and two digits that are not neighbours are no range either:
    # "hai tám ngày" is 28 days, "hai ba ngày" two or three.
    if len(words) <= 2 and words[-1] not in _SHORT_ONLY and not (_is_two_digit(words) and nxt not in _COUNTERS) \
            and not _is_context_numeral(words, prev, prev2, nxt, next2):
        return list(original)
    if all(w == "không" for w in words) and prev not in _ID_PREV:
        return list(original)
    # Three or four digits that count up or down ("một hai ba", "hai ba bốn người") are counting, not a code.
    if len(words) in (3, 4) and _is_counting(kind[1]) and not _is_context_numeral(words, prev, prev2, nxt, next2):
        return list(original)
    return [kind[1]]


_FRACTION_DIGITS = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8,
                    "chín": 9, "mười": 10}


def _join_fractions(tokens):
    """ "hai phần ba" -> "2/3", "một phần tư" -> "1/4": a small numerator below a small denominator."""
    out, i = [], 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1].lower() == "phần":
            a, b = _FRACTION_DIGITS.get(tokens[i].lower()), _FRACTION_DIGITS.get(tokens[i + 2].lower())
            after = tokens[i + 3].lower() if i + 3 < len(tokens) else ""
            before = out[-1].lower() if out else ""
            if a and b and a < b and after not in _NUMBER_WORDS and before not in _NUMBER_WORDS:
                out.append(f"{a}/{b}")
                i += 3
                continue
        out.append(tokens[i])
        i += 1
    return out


# Units spelled out after a number are written as symbols: "5 ki lô mét" -> "5 km". Longest first.
_UNITS = {
    "ki lô mét trên giờ": "km/h", "ki lô mét vuông": "km2", "ki lô mét": "km", "ki lô gam": "kg", "ki lô oát giờ": "kWh",
    "ki lô oát": "kW", "ki lô": "kg", "mét vuông": "m2", "mét khối": "m3", "xen ti mét vuông": "cm2",
    "xen ti mét": "cm", "mi li mét": "mm", "mi li lít": "ml", "mi li gam": "mg", "mi li giây": "ms",
    "mê ga oát": "MW", "mê ga bai": "MB", "mê ga bít": "Mb", "ghi ga bai": "GB", "ghi ga héc": "GHz", "tê ra bai": "TB",
    "mê ga héc": "MHz", "ki lô héc": "kHz", "héc ta": "ha", "oát": "W", "vôn": "V", "am pe": "A", "độ xê": "°C",
}
_UNIT_STARTS = {"mê ga", "ghi ga", "xen ti", "mi li", "tê ra", "am pe", "ki lô", "độ xê", "héc ta"}
_UNIT_RE = re.compile(r"(?<!\S)(\d+(?:[.,]\d+)?) (" + "|".join(sorted(_UNITS, key=len, reverse=True)).replace(" ", r"\s")
                      + r")(?!\S)", re.IGNORECASE)

# Southern contractions: "hăm ba" = hai mươi ba (23), "băm sáu" = ba mươi sáu (36).
_CONTRACTED = {"hăm": ["hai", "mươi"], "băm": ["ba", "mươi"]}


def _expand_contractions(tokens):
    out = []
    for k, t in enumerate(tokens):
        after = tokens[k + 1].lower() if k + 1 < len(tokens) else ""
        if t.lower() in _CONTRACTED and after in _DIGIT and after not in ("không", "linh", "lẻ", "năm"):
            out += _CONTRACTED[t.lower()]
        else:
            out.append(t)
    return out


# Names and idioms that contain number words ("Lầu Năm Góc" = the Pentagon). Their words are protected.
_FIXED = ("lầu năm góc", "năm châu bốn biển", "ba chìm bảy nổi", "chín người mười ý", "một sớm một chiều",
          "ba hoa", "ba phải", "ba que", "bảy nổi ba chìm", "năm thì mười họa", "một vừa hai phải", "mười phân vẹn mười",
          "trăm nghe không bằng một thấy", "ba mặt một lời", "tam sao thất bản", "một mất một còn", "chín suối",
          "mười mươi", "nhất cử lưỡng tiện", "một chín một mười", "năm bè bảy mối", "ba bề bốn bên")
_FIXED_RE = re.compile(r"(?<!\S)(" + "|".join(sorted(_FIXED, key=len, reverse=True)) + r")(?!\S)", re.IGNORECASE)
_PROTECT = "\u2060"  # word joiner: glued words are not number words while ITN runs


# A score is two numbers below 100 after a score word: "tỷ số sáu sáu" -> 6-6, "hòa một một" -> 1-1,
# "thắng hai mươi mốt mười chín" -> 21-19. Without this the two numbers read as one ("66"). After "thắng/thua/dẫn"
# a counter makes it a count range instead ("thắng hai ba trận" stays a range).
_UNIT_W = r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín)"
_BELOW_100 = rf"(?:(?:hai|ba|bốn|năm|sáu|bảy|bẩy|tám|chín) mươi(?: {_UNIT_W})?|mười(?: {_UNIT_W})?|{_UNIT_W})"
# Atomic groups: each number is read whole, so "hai mươi chín" is never split into 20 and 9 to make a score.
_SCORE_RE = re.compile(
    rf"(?i)(?<!\S)(tỷ số|tỉ số|hòa|hoà|(?:thắng|thua|dẫn)(?: với)?) ((?>{_BELOW_100})) ((?>{_BELOW_100}))(?!\S)")
_COUNT_AFTER = re.compile(r" (?:trận|bàn|điểm|ván|séc|set|game|lần|người|năm|tuổi|ngày|tháng|giờ|phút|giây|đội)\b",
                          re.I)


def _score_value(phrase: str) -> int:
    words = phrase.lower().split()
    return _DIGIT[words[0]] if len(words) == 1 and words[0] != "mười" else parse_number(words)


_TITLES = {"anh", "chị", "ông", "bà", "em", "cô", "chú", "bác", "cậu", "dì", "thầy", "bạn", "cháu", "con"}
_YEAR_AFTER = re.compile(r"\s*(?:qua|nay|trước|sau|ngoái|tới|rồi|liền|liên tiếp)\b", re.I)


def _scores(text: str) -> str:
    def rep(m):
        before = text[:m.start()].split()
        if m.group(1).lower() not in ("tỷ số", "tỉ số"):
            if before and before[-1].lower() in _TITLES:
                return m.group(0)  # "anh Thắng ba hai" is a name, not a score
            if _COUNT_AFTER.match(text, m.end()):
                return m.group(0)  # "thắng hai ba trận": a range of matches
        if m.group(3).lower().endswith("năm") and _YEAR_AFTER.match(text, m.end()):
            return m.group(0)  # "hòa hai năm qua": năm is the year noun
        return f"{m.group(1)} {_score_value(m.group(2))}-{_score_value(m.group(3))}"
    return _SCORE_RE.sub(rep, text)


def spoken_to_digits(text: str, fix_corrections: bool = True) -> str:
    """Rewrite spoken numbers in `text` as digits. `fix_corrections` applies "ba à không bốn" -> "4"."""
    text = _scores(text)
    text = _FIXED_RE.sub(lambda m: m.group(0).replace(" ", _PROTECT + " " + _PROTECT).join(_PROTECT * 2), text)
    return _spoken_to_digits(text, fix_corrections).replace(_PROTECT, "")


def _spoken_to_digits(text: str, fix_corrections: bool) -> str:
    tokens = _join_fractions(_join_decimals(_expand_contractions(text.split())))
    out, i = [], 0
    while i < len(tokens):
        w = tokens[i].lower()
        prev = out[-1].lower() if out else ""
        after = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        starts = w in _NUMBER_WORDS and w not in _NO_START
        if w == "tư" and prev in _TU_CONTEXT and after not in _TU_COMPOUND:
            starts = True
        if not starts:
            out.append(tokens[i])
            i += 1
            continue
        j, marked = _collect(tokens, i, fix_corrections)
        nxt = tokens[j].lower() if j < len(tokens) else ""
        next2 = tokens[j + 1].lower() if j + 1 < len(tokens) else ""
        if all(not isinstance(m, tuple) for m in marked):
            out += _render(marked, prev, out[-2].lower() if len(out) > 1 else "", nxt, next2, tokens[i:j])
        else:
            groups, fillers = _groups(marked)
            for g_idx, group in enumerate(groups):
                last = g_idx == len(groups) - 1
                g_prev = out[-1].lower() if out else ""
                g_prev2 = out[-2].lower() if len(out) > 1 else ""
                g_next = nxt if last else groups[g_idx + 1][0]
                g_next2 = next2 if last else (groups[g_idx + 1][1:2] or [""])[0]
                out += _render(group, g_prev, g_prev2, g_next, g_next2, group)
                if not last:
                    out.append(fillers[g_idx])
        i = j
    text = " ".join(out)
    # A single consonant letter and its number are one code: "b 2" -> "b2", "f 0" -> "f0".
    text = re.sub(r"(?<!\S)(?<!(?<!\S)\w )(?<!\d )([b-df-hj-np-tvwz]) (\d{1,2})(?!\S)", r"\1\2", text)
    text = re.sub(r"(?<![\d/])(\d+) (?:gạch )?(?:xẹt|xoẹt|xuyệt|xuộc|chéo) (\d+)\b", r"\1/\2", text)
    text = re.sub(r"\b(\d+) gạch (?:ngang|nối) (\d+)\b", r"\1-\2", text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_UNITS[' '.join(m.group(2).lower().split())]}", text)
    # A year range said short: "từ 2020 đến 25" -> 2020 đến 2025, "giai đoạn 2021 25" -> 2021-2025.
    def year_range(m):
        first, sep, short = m.group(1), m.group(2), m.group(3)
        if int(short) <= int(first[2:]):
            return m.group(0)
        # Without "đến", only after a term word: "năm 2020 25 người" is a year and a count.
        if sep == " " and not re.search(r"(?:đoạn|kỳ|kì|khóa|khoá|niên)\s$", m.string[:m.start()]):
            return m.group(0)
        full = first[:2] + short
        return f"{first}-{full}" if sep == " " else f"{first}{sep}{full}"
    text = re.sub(r"(?<!\S)((?:19|20)\d\d)( đến | tới |-| )(\d\d)(?![\d,.%]|\S)", year_range, text)
    # Negative numbers: "âm ba độ" -> "-3 độ".
    text = re.sub(r"(?i)(?<!\S)âm (\d)", r"-\1", text)
    # Decimals and percent: "3 phẩy 5" -> "3,5", "15 phần trăm" -> "15%".
    text = re.sub(r"(?<![\d,])\b(\d+) phẩy (\d+)\b", r"\1,\2", text)
    text = re.sub(r"\b(\d+(?:[.,]\d+)?) phần trăm\b", r"\1%", text)
    return text


# ------------------------------------------------------------------------------------------------------- decimals
def _number_value(words):
    """Integer for a run that is a full reading or a digit-by-digit reading ("không" alone is 0), else None."""
    if words == ["không"]:
        return "0"
    value = parse_number(words)
    if value is not None:
        return str(value)
    return _digit_string(words) if words else None


def _join_decimals(tokens):
    """Rewrite "<number> phẩy <number> [scale]" as "<int>,<frac> [scale]": "ba phẩy bốn triệu" -> "3,4 triệu".

    The scale word is kept, since "3,4 triệu" is how Vietnamese writes it (3,4000000 would be wrong).
    """
    out, i = [], 0
    frac_words = set(_DIGIT) | {"mười", "mươi", "trăm"}
    while i < len(tokens):
        if tokens[i].lower() not in ("phẩy", "chấm") or not out:
            out.append(tokens[i])
            i += 1
            continue
        # Integer part: the number words just before "phẩy", without a trailing scale word.
        k = len(out)
        while k > 0 and out[k - 1].lower() in _NUMBER_WORDS:
            k -= 1
        # The longest run before "phẩy" that reads as one number: "chín ngàn chín trăm mười ba phẩy ..." = 9913,
        # but "ba triệu | hai phẩy năm" keeps "ba triệu" apart when the whole does not read.
        while k < len(out) and _number_value([t.lower() for t in out[k:]]) is None:
            k += 1
        j = i + 1
        while j < len(tokens) and tokens[j].lower() in frac_words:
            j += 1
        # "hai chấm không ba chấm không" is 2.0 and 3.0: the last digit starts the next decimal.
        if j - i > 2 and j < len(tokens) and tokens[j].lower() in ("phẩy", "chấm"):
            j -= 1
        # "ba phẩy bốn năm" is 3,4 years: a final "năm" after a digit is the noun, not a digit.
        if j - i == 3 and tokens[j - 1].lower() == "năm" and (j == len(tokens) or tokens[j].lower() not in _NUMBER_WORDS):
            j -= 1
        left = _number_value([t.lower() for t in out[k:]])
        right = _number_value([t.lower() for t in tokens[i + 1:j]])
        if left is None or right is None:
            out.append(tokens[i])
            i += 1
            continue
        mark = "," if tokens[i].lower() == "phẩy" else "."  # "hai chấm không" -> 2.0 (versions, ratings)
        out[k:] = [f"{left}{mark}{right}"]
        i = j
    return out
