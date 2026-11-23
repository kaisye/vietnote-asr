from vnasr.hotwords import load


def _hw(tmp_path, text):
    p = tmp_path / "hw.txt"
    p.write_text(text, encoding="utf-8")
    return load(p)


def test_names_and_readings(tmp_path):
    hw = _hw(tmp_path, "# c\nNvidia | en vi đi a, en vi đi\nJensen Huang\nVinFast | vin phát\nX | em\n")
    assert hw.apply("Ông jensen huang điều hành en vi đi a.") == "Ông Jensen Huang điều hành Nvidia."
    assert hw.apply("Xe Vin phát và nvidia") == "Xe VinFast và Nvidia"
    assert hw.apply("anh em vin phátx") == "anh em vin phátx"  # one-syllable readings are ignored; whole words only
    assert "nvidia" in hw.boost_phrases() and "vin phát" in hw.boost_phrases()
