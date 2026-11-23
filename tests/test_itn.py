import pytest

from vnasr.itn import parse_number, spoken_to_digits
from vnasr.numbers import spell_integer


@pytest.mark.parametrize("n", [*range(10, 1200), 1900, 1975, 2000, 2005, 2010, 2024, 2025, 2026, 10_000, 15_500,
                               100_000, 250_000, 1_000_000, 1_250_000, 3_000_000_000])
def test_round_trip_with_spell_integer(n):
    assert parse_number(spell_integer(n).split()) == n


@pytest.mark.parametrize("spoken, written", [
    ("năm hai không hai lăm", "năm 2025"),
    ("năm hai nghìn không trăm hai mươi lăm", "năm 2025"),
    ("năm hai không hai lăm là một năm đặc biệt", "năm 2025 là một năm đặc biệt"),
    ("sinh năm hai không không năm", "sinh năm 2005"),
    ("sinh năm hai nghìn không trăm linh năm", "sinh năm 2005"),
    ("số điện thoại không chín bốn hai ba bốn năm sáu bảy", "số điện thoại 094234567"),
    ("đến năm hai nghìn không trăm hai mươi lăm thì xong", "đến năm 2025 thì xong"),
    ("ngày mười lăm tháng mười một", "ngày 15 tháng 11"),
    ("họp lúc bảy giờ ba mươi", "họp lúc 7 giờ 30"),
    ("tăng ba phẩy năm phần trăm", "tăng 3,5%"),
    ("mười lăm phần trăm", "15%"),
    ("giá năm mươi nghìn đồng", "giá 50.000 đồng"),
    ("hai mươi năm sau", "20 năm sau"),
    ("một trăm năm cô đơn", "100 năm cô đơn"),
    ("hai ba người", "hai ba người"),
    ("có một người", "có một người"),
    ("thứ hai tuần sau", "thứ hai tuần sau"),
    ("không phải vậy", "không phải vậy"),
    ("năm nay", "năm nay"),
    ("tháng hai", "tháng 2"),
    ("tháng mười hai năm hai nghìn không trăm mười ba", "tháng 12 năm 2013"),
    ("tháng mười một và tháng mười hai năm hai nghìn không trăm mười ba tăng gần một năm",
     "tháng 11 và tháng 12 năm 2013 tăng gần một năm"),
    ("mười hai năm hai nghìn không trăm mười ba", "12 năm 2013"),
    ("từ bốn phẩy không cho đến", "từ 4,0 cho đến"),
    ("tôi không phẩy gì", "tôi không phẩy gì"),
    # Decimals before a scale word keep the scale word (2026-11-21).
    ("ba phẩy bốn triệu chiếc", "3,4 triệu chiếc"),
    ("một phẩy hai tỷ đồng", "1,2 tỷ đồng"),
    ("mười hai phẩy bảy lăm triệu", "12,75 triệu"),
    ("không phẩy năm", "0,5"),
    ("ba triệu bốn trăm nghìn chiếc", "3,4 triệu chiếc"),
    # Shortened readings (2026-11-21).
    ("hôm qua tao chuyển mày ba triệu tư rồi nha", "hôm qua tao chuyển mày 3,4 triệu rồi nha"),
    ("ba triệu rưỡi", "3,5 triệu"),
    ("mười lăm triệu tám", "15,8 triệu"),
    ("hai nghìn rưỡi", "2.500"),
    ("hai giờ rưỡi", "2 giờ rưỡi"),
    ("ba triệu năm ánh sáng", "3 triệu năm ánh sáng"),
    # Self-correction inside a number stays as spoken instead of "1947 mươi lăm".
    # ... and the restart is repaired: "bốn" was abandoned for "bảy mươi lăm" (2026-11-21 rewrite).
    ("năm ơ một nghìn chín trăm bốn bảy mươi lăm ngày ba mươi tháng tư á", "năm ơ 1975 ngày 30 tháng 4 á"),
    # Question particle "không" after a number.
    ("vậy năm hai nghìn không trăm hai mươi mốt không", "vậy năm 2021 không"),
    ("hai mươi mốt không", "21 không"),
    # Phone numbers read in groups.
    ("đọc liền một dãy nè không tám mươi sáu hai trăm hai mươi hai ba trăm năm mươi hai", "đọc liền một dãy nè 086222352"),
    ("không chín mươi tám hai trăm ba mươi tư năm trăm sáu mươi bảy", "098234567"),
    ("không có hai mươi người", "không có 20 người"),
    ("hai tỷ năm trăm triệu đồng", "2,5 tỷ đồng"),
    ("một trăm hai mươi triệu", "120 triệu"),
    ("ba triệu bốn trăm nghìn năm trăm đồng", "3.400.500 đồng"),
    ("giá vàng một trăm hai mươi bảy phẩy năm triệu đồng", "giá vàng 127,5 triệu đồng"),
    ("ngày ba mươi tháng tư năm một nghìn chín trăm bảy mươi lăm", "ngày 30 tháng 4 năm 1975"),
    ("hai mươi lăm tuổi", "25 tuổi"),
    # Product and version names (2026-11-21).
    ("model này nó sẽ chạy trên raspberry pi bốn", "model này nó sẽ chạy trên raspberry pi 4"),
    ("mua iphone mười lăm pro max", "mua iphone 15 pro max"),
    ("cài windows mười một", "cài windows 11"),
    ("con chip snapdragon tám", "con chip snapdragon 8"),
    ("react một cái là xong", "react một cái là xong"),
])
def test_spoken_to_digits(spoken, written):
    assert spoken_to_digits(spoken) == written


@pytest.mark.parametrize("spoken, written", [
    ("hạ tầng một cách", "hạ tầng một cách"),
    ("đất chiếm một phần", "đất chiếm một phần"),
    ("trong một trăm bốn lăm ngày", "trong 145 ngày"),
    ("chuyển đổi số không chỉ", "chuyển đổi số không chỉ"),
    ("tuyến đường hai mét", "tuyến đường 2 mét"),
    ("năm phần trăm", "5%"),
])
def test_false_positives_from_dev(spoken, written):
    assert spoken_to_digits(spoken) == written


# 2026-11-21 rewrite: segmentation, restarts, fillers, colloquial readings, context rules.
@pytest.mark.parametrize("spoken, written", [
    # Several numbers in one run.
    ("sinh năm hai nghìn không trăm mười sáu hai nghìn không trăm mười tám", "sinh năm 2016 2018"),
    ("hai mươi lăm triệu ba mươi triệu", "25 triệu 30 triệu"),
    ("còn hai mốt ba mươi và", "còn 21 30 và"),
    ("mười tám mười chín hai mươi", "18 19 20"),
    # Ranges.
    ("độ tám chín triệu", "độ 8-9 triệu"),
    ("này là ba lăm bốn mươi nghìn một cân", "này là 35-40 nghìn một cân"),
    ("hai ba người", "hai ba người"),
    ("điểm gleason hai đến bốn", "điểm gleason 2 đến 4"),
    # Restarts, stutters, fillers, corrections.
    ("năm một nghìn chín trăm bốn bảy mươi lăm", "năm 1975"),
    ("hai hai mươi người", "20 người"),
    ("năm hai nghìn ờ không trăm hai mươi hai", "năm 2022"),
    ("hai nghìn ờ ờ không trăm mười hai", "2012"),
    ("ba triệu ừm bốn trăm nghìn", "3,4 triệu"),
    ("chuyển ba à không bốn triệu", "chuyển 4 triệu"),
    ("hai ờ ba người", "hai ờ ba người"),
    # Colloquial readings.
    ("hai lăm tuổi", "25 tuổi"),
    ("đâu có bốn lăm ngày", "đâu có 45 ngày"),
    ("hai tám độ", "28 độ"),
    ("một trăm hai nghìn", "120.000"),
    ("hai nghìn mốt", "2100"),
    ("hai chục ngàn", "20.000"),
    ("mười một nghìn sáu trăm tỷ đồng", "11.600 tỷ đồng"),
    ("năm hai nghìn hai hai", "năm 2022"),
    ("trong năm hai nghìn không trăm hai hai", "trong năm 2022"),
    ("nhìn đến hai không năm mươi", "nhìn đến 2050"),
    ("năm hai nghìn không trăm mười bảy tỷ lệ", "năm 2017 tỷ lệ"),
    ("hai phần ba trường", "2/3 trường"),
    # "năm" as the year noun, not five.
    ("tháng năm năm nay", "tháng 5 năm nay"),
    ("tháng ba năm vừa rồi", "tháng 3 năm vừa rồi"),
    ("cùng kỳ năm trước", "cùng kỳ năm trước"),
    ("khoảng hai ba năm", "khoảng hai ba năm"),
    ("thêm một năm đồng thời", "thêm một năm đồng thời"),
    ("chi phí là một nghìn một năm", "chi phí là 1000 một năm"),
    ("khoảng ba phẩy bốn năm nay", "khoảng 3,4 năm nay"),
    ("ngày tháng năm sinh", "ngày tháng năm sinh"),
    # Context: words that are not Vietnamese, counters, compounds.
    ("model này chạy trên raspberry pi bốn", "model này chạy trên raspberry pi 4"),
    ("chạy python ba", "chạy python 3"),
    ("writing task hai", "writing task 2"),
    ("react một cái là xong", "react một cái là xong"),
    ("team năm người", "team năm người"),
    ("highlight một số cái", "highlight một số cái"),
    ("bằng b hai", "bằng b2"),
    ("có mười g chín gam", "có 10 g 9 gam"),
    ("là một đồng chí tốt", "là một đồng chí tốt"),
    ("mỗi ngày một lần", "mỗi ngày một lần"),
    ("bảy tháng bảy thì", "bảy tháng 7 thì"),
    ("ngày một phát triển", "ngày một phát triển"),
    ("thông tư bốn mươi sáu năm hai nghìn không trăm mười tám", "thông tư 46 năm 2018"),
    ("nghị định không bảy", "nghị định 07"),
    ("giai đoạn hai thì", "giai đoạn 2 thì"),
    ("cung cấp một cơ hội", "cung cấp một cơ hội"),
    ("bất kỳ một công ty", "bất kỳ một công ty"),
    ("không không không hiểu", "không không không hiểu"),
    ("mọi năm tỷ lệ", "mọi năm tỷ lệ"),
    ("sẽ là hai triệu chứng", "sẽ là hai triệu chứng"),
    ("lúc mười bốn giờ tám chiều", "lúc 14 giờ 8 chiều"),
    ("lần thứ hai mươi", "lần thứ 20"),
    ("một hai ba", "một hai ba"),
    ("đến hai không hai lăm", "đến 2025"),
    ("mã ba hai năm hai", "mã 3252"),
    ("ngày hăm ba tháng chạp", "ngày 23 tháng chạp"),
    # From the VietAI spoken_norm validation split (2026-11-22).
    ("ba nghìn tám", "3.800"),
    ("sáu tư nghìn sáu trăm", "64.600"),
    ("năm triệu không ngàn không trăm linh sáu", "5.000.006"),
    ("chín ngàn chín trăm mười ba phẩy năm mươi tư mi li mét", "9913,54 mm"),
    ("ba mươi phẩy không không năm", "30,005"),
    ("năm một nghìn ba trăm ba ba", "năm 1333"),
    ("không năm một bốn tám không ba bốn một sáu ba năm", "051480341635"),
    ("chín không không", "900"),
    ("bảy giờ ba nhăm một trong", "7 giờ 35 một trong"),
    ("hẻm mười hai xẹt ba", "hẻm 12/3"),
    ("số nhà ba trăm xoẹt hai", "số nhà 300/2"),
    ("nhiệt độ âm ba độ", "nhiệt độ -3 độ"),
    ("âm thanh hai chiều", "âm thanh hai chiều"),
    ("phiên bản hai chấm không", "phiên bản 2.0"),
    ("chấm điểm ba bạn", "chấm điểm ba bạn"),
    ("dài năm ki lô mét", "dài 5 km"),
    ("căn hộ năm mươi mét vuông", "căn hộ 50 m2"),
    ("tốc độ một trăm hai mươi ki lô mét trên giờ", "tốc độ 120 km/h"),
    ("ram tám ghi ga bai", "ram 8 GB"),
    ("ông mười với giá", "ông mười với giá"),
    ("anh tư nói", "anh tư nói"),
    ("ba trăm bốn ba euro năm từ", "343 euro năm từ"),
    ("trên hai trăm chín mươi héc ta còn", "trên 290 ha còn"),
    ("là khoảng một trăm ba mươi nghìn một cân", "là khoảng 130.000 một cân"),
    ("lương hai triệu một tháng", "lương 2 triệu một tháng"),
    ("thế hệ hai chấm không ba chấm không", "thế hệ 2.0 3.0"),
    ("giá ba nghìn tám một ký", "giá 3.800 một ký"),
    # From the 36-minute news video reviewed by the user (2026-11-22).
    ("trong một trăm hai mươi năm một trăm hai mươi năm đấy", "trong 120 năm 120 năm đấy"),
    ("đúng không chín trăm bốn trăm chín mươi ba đúng không", "đúng không 900 493 đúng không"),
    ("bị lầu năm góc đưa vào", "bị lầu năm góc đưa vào"),
    ("khắp năm châu bốn biển", "khắp năm châu bốn biển"),
    ("từ hai nghìn không trăm hai mươi đến hai mươi lăm thì", "từ 2020 đến 2025 thì"),
    ("giai đoạn hai ngàn hai mốt hai lăm thì", "giai đoạn 2021-2025 thì"),
    ("năm hai nghìn không trăm hai mươi có hai mươi người", "năm 2020 có 20 người"),
    ("thì sau đó đến năm hai mươi tư đấy thì", "thì sau đó đến năm 24 đấy thì"),
    ("năm hai mươi người", "năm hai mươi người"),
    ("năm trăm hai mươi", "520"),
    ("cho đấy là bốn tỷ tám là tối thiểu", "cho đấy là 4,8 tỷ là tối thiểu"),
    ("mua hết hai trăm rưỡi nghìn", "mua hết 250.000"),
    ("băm sáu phố phường", "36 phố phường"),
])
def test_rewrite_2026_11_21(spoken, written):
    assert spoken_to_digits(spoken) == written


def test_corrections_can_be_turned_off():
    assert spoken_to_digits("chuyển ba à không bốn triệu", fix_corrections=False) == "chuyển ba à không 4 triệu"


@pytest.mark.parametrize("word, foreign", [("pi", True), ("iphone", True), ("task", True), ("rtx", True),
                                           ("max", True), ("be", False), ("ba", False), ("người", False),
                                           ("chat", False), ("gen", False)])
def test_is_foreign(word, foreign):
    from vnasr.itn import is_foreign
    assert is_foreign(word) is foreign


@pytest.mark.parametrize("spoken, written", [
    ("sau khi đạt tỷ số sáu sáu", "sau khi đạt tỷ số 6-6"),
    ("tỷ số hai mươi mốt mười chín", "tỷ số 21-19"),
    ("thắng ba không", "thắng 3-0"),
    ("hòa không không", "hòa 0-0"),
    ("dẫn với tỷ số hai một", "dẫn với tỷ số 2-1"),
    ("thua mười chín hai mươi mốt", "thua 19-21"),
    ("thắng hai ba trận", "thắng hai ba trận"),  # counter: a range of matches, not a score
    ("hòa hai năm qua", "hòa hai năm qua"),
    ("anh thắng ba hai", "anh thắng ba hai"),  # a name
    ("tỷ số sáu", "tỷ số 6"),
])
def test_scores_2026_11_22(spoken, written):
    assert spoken_to_digits(spoken) == written


@pytest.mark.parametrize("spoken, written", [
    ("có được tỉ số mười lăm mười bốn điểm", "có được tỉ số 15-14 điểm"),
    ("phan viết thắng hai mươi chín đến hai mươi lăm tuổi", "phan viết thắng 29 đến 25 tuổi"),
    ("lượn lờ tỉ số hai mươi tám với", "lượn lờ tỉ số 28 với"),
])
def test_scores_numbers_read_whole(spoken, written):
    assert spoken_to_digits(spoken) == written


@pytest.mark.parametrize("spoken, written", [
    ("gần hai trăm ngàn chữ ký", "gần 200.000 chữ ký"),
    ("bán được ba mươi tám nghìn xe", "bán được 38.000 xe"),
    ("năm một nghìn năm trăm ba mươi bảy", "năm 1537"),           # 1000–2100: years stay plain
    ("năm hai nghìn không trăm hai mươi lăm", "năm 2025"),
    ("khoảng năm nghìn người", "khoảng 5.000 người"),
    ("ba triệu bốn trăm nghìn năm trăm", "3.400.500"),
])
def test_thousand_separators_2026_11_23(spoken, written):
    assert spoken_to_digits(spoken) == written
