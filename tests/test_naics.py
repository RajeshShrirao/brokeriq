from brokeriq.tools.naics import lookup_naics


def test_naics_lookup_software():
    result = lookup_naics("Acme Analytics", "saas")
    assert result is not None
    assert result["code"] == "5415"


def test_naics_lookup_trucking():
    result = lookup_naics("Fast Freight Lines")
    assert result is not None
    assert result["code"] == "48"


def test_naics_lookup_unknown():
    assert lookup_naics("Qwerty Zxcvbn Co") is None


def test_naics_lookup_empty():
    assert lookup_naics("") is None
