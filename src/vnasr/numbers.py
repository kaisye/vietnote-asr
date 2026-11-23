"""Spell numbers the way Vietnamese speakers read them.

Readings vary by region (nghìn/ngàn, linh/lẻ, tư/bốn). This module fixes one
form so that labels and scoring agree; it does not claim the others are wrong.
"""

import re

_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_SCALES = ((10**9, "tỷ"), (10**6, "triệu"), (10**3, "nghìn"))
_MAX_SPELLED = 10**12

# (?![.,]\d) stops a partial match such as "50" out of "50.000đ" when the full token is followed by a letter.
NUMBER_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])(?![.,]\d)(\s?%)?")
_NUM = r"\d+(?:[.,]\d{3})*"
_MONEY_RE = re.compile(rf"(?<![\w.,])({_NUM})\s?(vnđ|vnd|đồng|đ)(?!\w)")
_SHORT_SCALE_RE = re.compile(rf"(?<![\w.,])({_NUM}(?:,\d{{1,2}})?)\s?(k|tr|ty|tỷ|triệu|nghìn|ngàn)(?!\w)")
_CLOCK_RE = re.compile(r"(?<![\w.,:])([01]?\d|2[0-4])\s?(?:h|:)\s?([0-5]\d)?(?![\w:])")
_SLASH_RE = re.compile(r"(?<![\w./])(\d{1,4})/(\d{1,3}(?:\.\d{3})+|\d{1,4})(?:/(\d{4}))?(?![\w/])")
# "15/11" is a date only after a time word ("ngày 15/11", "hạn 15/11") or with a year; otherwise it is an
# address, ratio or score and reads "mười lăm trên mười một" ("số 15/11 Lê Duẩn", "tỉ lệ 3/4").
_DATE_CUE_RE = re.compile(r"(?:ngày|hôm|nay|mai|mùng|mồng|sáng|trưa|chiều|tối|đêm|từ|đến|tới|hạn|hạn chót|deadline|trước|sau|vào|"
                          r"lễ|dịp|đợt|thứ\s+\w+|chủ\s+nhật)\s*[,:]?\s*$")
_SHORT_SCALE = {"k": "nghìn", "tr": "triệu", "ty": "tỷ"}
_RANGE_RE = re.compile(r"(?<![\w.,:])(\d[\d:.,h]*)\s*[–—]\s*(\d[\d:.,h]*)(?![\w:])")


def _below_hundred(value: int) -> list[str]:
    tens, units = divmod(value, 10)
    if tens == 0:
        return [_DIGITS[units]]
    words = ["mười"] if tens == 1 else [_DIGITS[tens], "mươi"]
    if units == 0:
        return words
    if units == 5:
        words.append("lăm")
    elif units == 1 and tens > 1:
        words.append("mốt")
    elif units == 4 and tens > 1:
        words.append("tư")
    else:
        words.append(_DIGITS[units])
    return words


def _below_thousand(value: int, full: bool) -> list[str]:
    """Read one three-digit group; `full` keeps "không trăm" inside larger numbers."""
    hundreds, rest = divmod(value, 100)
    if hundreds == 0 and not full:
        return _below_hundred(rest)
    words = [_DIGITS[hundreds], "trăm"]
    if rest == 0:
        return words
    if rest < 10:
        return [*words, "linh", _DIGITS[rest]]
    return [*words, *_below_hundred(rest)]


def spell_integer(value: int) -> str:
    if value == 0:
        return _DIGITS[0]
    words: list[str] = []
    remaining = value
    for scale, name in _SCALES:
        group, remaining = divmod(remaining, scale)
        if group:
            words += [*_below_thousand(group, full=bool(words)), name]
    if remaining:
        words += _below_thousand(remaining, full=bool(words))
    return " ".join(words)


def _spell_digits(digits: str) -> str:
    return " ".join(_DIGITS[int(digit)] for digit in digits)


def _spell_part(digits: str) -> str:
    # Leading zeros and very long runs are codes or phone numbers, read digit by digit.
    if (len(digits) > 1 and digits[0] == "0") or int(digits) >= _MAX_SPELLED:
        return _spell_digits(digits)
    return spell_integer(int(digits))


def spell_number(token: str) -> str:
    """Spell one numeric token such as 15, 1.000, 3,5 or 1.5.2."""
    parts = re.split(r"[.,]", token)
    if len(parts) == 1:
        return _spell_part(token)
    # Vietnamese groups thousands with dots, and three-digit groups read the same with commas.
    if parts[0] != "0" and not parts[0].startswith("0") and all(len(part) == 3 for part in parts[1:]):
        return _spell_part("".join(parts))
    if len(parts) == 2:
        return f"{_spell_part(parts[0])} phẩy {_spell_part(parts[1])}"
    return " chấm ".join(_spell_part(part) for part in parts)


def _spell_units(text: str) -> str:
    """Read money, short scales (50k, 2tr), clock times (7h30, 20:30) and dates (15/11, 15/11/2026)."""
    text = _MONEY_RE.sub(lambda m: f"{spell_number(m.group(1))} đồng", text)
    text = _SHORT_SCALE_RE.sub(
        lambda m: f"{spell_number(m.group(1))} {_SHORT_SCALE.get(m.group(2), m.group(2))}", text)

    def clock(m: re.Match) -> str:
        hour, minute = int(m.group(1)), m.group(2)
        spoken = f"{spell_integer(hour)} giờ"
        return f"{spoken} {spell_integer(int(minute))}" if minute and int(minute) else spoken

    # A dash between two times or numbers is read "đến": "10:00 – 11:00", "25-30 năm".
    text = _RANGE_RE.sub(r"\1 đến \2", text)
    text = _CLOCK_RE.sub(clock, text)
    # "12:00 giờ GMT" already says "giờ" once the clock is spelled.
    text = re.sub(r"\bgiờ(?:\s+(?:không|0))?\s+giờ\b", "giờ", text)

    def slash(m: re.Match) -> str:
        if "." in m.group(2):  # "1/100.000": a ratio
            return f"{_spell_part(m.group(1))} trên {spell_number(m.group(2))}"
        a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
        # "01/04" is written as a date; "30/4 1/5" follows a date.
        padded = len(m.group(1)) == 2 and len(m.group(2)) == 2 and m.group(1)[0] == "0" or m.group(2)[0] == "0"
        after_date = re.search(r"\d/\d{1,2}\s*[-–,]?\s*$", m.string[:m.start()])
        is_date = 1 <= a <= 31 and 1 <= b <= 12 and (year or padded or after_date
                                                     or _DATE_CUE_RE.search(m.string[:m.start()]))
        if not is_date:
            return f"{_spell_part(m.group(1))} trên {_spell_part(m.group(2))}" + (f" trên {_spell_part(year)}" if year else "")
        spoken = f"{spell_integer(a)} tháng {spell_integer(b)}"
        return f"{spoken} năm {spell_integer(int(year))}" if year else spoken

    return _SLASH_RE.sub(slash, text)


def spell_numbers_in_text(text: str) -> str:
    def replace(match: re.Match) -> str:
        spoken = spell_number(match.group(0).rstrip("% "))
        return f"{spoken} phần trăm" if match.group(1) else spoken

    text = _spell_identifiers(text)
    text = _spell_math(_unpad_dates(text))
    text = _spell_units(text)
    text = _spell_codes(text)
    return _fix_ordinals(NUMBER_RE.sub(replace, text))


# "mùng 05 tháng 03" is read "mùng năm tháng ba": a zero-padded day, month or hour is a plain number.
_PADDED_RE = re.compile(r"(?i)(\b(?:ngày|mùng|mồng|tháng|lúc|giờ)\s+)0(\d)\b")


def _unpad_dates(text: str) -> str:
    return _PADDED_RE.sub(r"\1\2", text)


# "1 + 1 = 2" is read "một cộng một bằng hai".
_MATH_RE = re.compile(r"(?<=\d)\s*([+=×÷])\s*(?=\d)")
_MATH_WORDS = {"+": "cộng", "=": "bằng", "×": "nhân", "÷": "chia"}


def _spell_math(text: str) -> str:
    return _MATH_RE.sub(lambda m: f" {_MATH_WORDS[m.group(1)]} ", text)


# Letters glued to digits ("G8", "4G", "H1N1", "CD4", "VN30", "COVID-19") are read letter by letter with the digits
# as a number: "g tám", "bốn g", "h một n một". Patterns read in English ("B2B", "1st") are left alone, so the
# sentence keeps a digit and can be filtered out of the labels.
_CODE_RE = re.compile(r"(?<![\w.,])(?=[\w-]*\d)(?=[\w-]*[^\W\d_])([^\W_]+(?:-[^\W_]+)*)(?![\w])")
_ENGLISH_CODE_RE = re.compile(r"(?i)^(?:[a-z]2[a-z]+|\d+(?:st|nd|rd|th)|[a-z]\d[a-z]\d?[a-z]?\d?e+)$")


def _spell_codes(text: str) -> str:
    def code(m: re.Match) -> str:
        token = m.group(1)
        if _ENGLISH_CODE_RE.match(token) or re.fullmatch(r"\d+(?:[.,]\d+)?", token):
            return token
        parts = re.findall(r"\d+|[^\W\d_]+", token)
        if not parts or any(p.isdigit() and len(p) > 4 for p in parts):
            return token
        return " ".join(_spell_part(p) if p.isdigit() else p for p in parts)

    return _CODE_RE.sub(code, text)


# Identifiers are read digit by digit ("ba hai năm hai"), never as amounts ("ba nghìn hai trăm năm mươi hai").
# Labels that spelled them as amounts taught the recogniser to invent "mươi"/"nghìn" (2026-11-21).
_ID_CUE = (r"(?:số\s+điện\s+thoại|điện\s+thoại|sđt|đt|hotline|zalo|số\s+tài\s+khoản|tài\s+khoản|stk|mã\s+số|mã|otp|"
           r"pin|code|cmnd|cccd|căn\s+cước|biển\s+số|số\s+hiệu|số\s+thẻ|số\s+hợp\s+đồng|số\s+đơn|mã\s+đơn)")
_ID_GROUPS = r"\d+(?:[ .\-]\d+)*"
_CUED_ID_RE = re.compile(rf"(?i)(?<!\w)({_ID_CUE}\s*(?:là|:)?\s*)({_ID_GROUPS})(?![\w])")
# A phone number written in groups: "086 222 3252", "0912.345.678", "028-3823-4567".
_PHONE_RE = re.compile(r"(?<![\w.,])0\d{2,4}(?:[ .\-]\d{3,4}){1,3}(?![\w])(?![.,]\d)")
# A long bare run with no unit after it is a code or account number, not an amount ("98000000 người" stays).
_LONG_RUN_RE = re.compile(r"(?<![\w.,])\d{8,}(?![\w])(?![.,]\d)(?!\s*(?:%|đồng|vnđ|vnd|usd|đô|người|dân|lượt|km|m|kg|tấn|lít))")


def _spell_identifiers(text: str) -> str:
    digits_only = lambda s: _spell_digits(re.sub(r"\D", "", s))
    text = _CUED_ID_RE.sub(lambda m: m.group(1) + digits_only(m.group(2)), text)
    text = _PHONE_RE.sub(lambda m: digits_only(m.group(0)), text)
    return _LONG_RUN_RE.sub(lambda m: _spell_digits(m.group(0)), text)


# A bare 4 after "tháng"/"thứ" is read "tư" (tháng tư, thứ tư, lần thứ tư), and a bare 1 after "thứ" is read
# "nhất" (lần thứ nhất). "thứ bốn mươi" and "tháng một" stay as they are.
_NOT_COMPOUND = r"(?!\s+(?:mươi|trăm|nghìn|ngàn|triệu|tỷ|lăm|linh|lẻ)\b)"
_ORDINAL_FIXES = (
    (re.compile(rf"\b(tháng|thứ)\s+bốn\b{_NOT_COMPOUND}"), r"\1 tư"),
    (re.compile(rf"\b(thứ)\s+một\b{_NOT_COMPOUND}"), r"\1 nhất"),
)


def _fix_ordinals(text: str) -> str:
    for pattern, repl in _ORDINAL_FIXES:
        text = pattern.sub(repl, text)
    return text
