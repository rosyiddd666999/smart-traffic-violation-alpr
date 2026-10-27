import pytest

from alpr import PlateResult, _consensus, format_plate, normalize_plate_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B 1234 XYZ", "B1234XYZ"),
        ("b-1234-xyz", "B1234XYZ"),
        ("B.1234.XYZ", "B1234XYZ"),
        ("  B1234XYZ  ", "B1234XYZ"),
        ("81234XYZ", "B1234XYZ"),
        ("B1234XY2", "B1234XYZ"),
        ("B1O34XYZ", "B1034XYZ"),
        ("D 12 AB", "D12AB"),
        ("AB1234CD", "AB1234CD"),
        ("4AB 2345", "4AB2345"),
        ("4ab-2345", "4AB2345"),
        ("", ""),
        ("   ", ""),
        # Teks non-plat harus ditolak, bukan dipaksa menjadi plat.
        ("hello", ""),
        ("MAN", ""),
        ("ZAKO", ""),
        ("1234", ""),
    ],
)
def test_normalize_plate_text(raw, expected):
    assert normalize_plate_text(raw) == expected


def test_format_plate():
    assert format_plate("B1234XYZ") == "B 1234 XYZ"
    assert format_plate("D12AB") == "D 12 AB"
    assert format_plate("4AB2345") == "4AB 2345"
    assert format_plate("TIDAK TERBACA") == "TIDAK TERBACA"
    assert format_plate("") == ""


def _reading(text: str, confidence: float) -> PlateResult:
    return PlateResult(
        license_plate=text,
        confidence=confidence,
        source="real",
        engine="test",
    )


def test_consensus_votes_per_character():
    readings = [
        _reading("4AB2345", 0.99),
        _reading("4AB1345", 0.98),
        _reading("4AB2345", 0.97),
        _reading("4AB2345", 0.96),
        _reading("8AB2345", 0.95),
    ]
    result = _consensus(readings)
    assert result is not None
    assert result.license_plate == "4AB2345"
    assert result.agreement == pytest.approx(3 / 5, abs=0.01)


def test_consensus_rejects_without_whole_frame_support():
    # Voting per karakter bisa membentuk teks yang tidak pernah muncul utuh.
    readings = [
        _reading("AAW1444", 0.50),
        _reading("ZAW1444", 0.50),
        _reading("AA41444", 0.50),
        _reading("A44I444", 0.50),
        _reading("AKW1444", 0.50),
        _reading("AAW1449", 0.50),
    ]
    result = _consensus(readings)
    assert result is not None
    assert result.agreement < 0.34


def test_consensus_returns_none_without_valid_readings():
    assert _consensus([]) is None
    assert _consensus([_reading("", 0.0)]) is None
