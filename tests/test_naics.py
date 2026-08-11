from importlib import resources

from brokeriq.tools.naics import _load_codes, lookup_naics


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


def test_dataset_exists_as_package_resource():
    dataset = resources.files("brokeriq.data").joinpath("naics.csv")
    assert dataset.is_file()


def test_load_codes_via_packaged_path():
    rows = _load_codes()
    assert len(rows) == 25
    for row in rows:
        assert set(row) == {"code", "label", "keywords"}
    assert rows[0]["code"] == "11"
    assert rows[0]["label"] == "Agriculture / Farming"
