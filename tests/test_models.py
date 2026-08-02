from brokeriq.config import get_settings
from brokeriq.models import LeadInput, QualificationResult


def test_default_settings_load():
    settings = get_settings()
    assert settings.model.startswith("openrouter")
    assert settings.env in {"dev", "test", "prod"}


def test_lead_input_defaults():
    lead = LeadInput(company_name="Acme Widgets")
    assert lead.revenue_band == "unknown"
    assert lead.notes == ""


def test_lead_input_rejects_empty_name():
    import pytest

    with pytest.raises(ValueError):
        LeadInput(company_name="")  # type: ignore[arg-type]


def test_qualification_score_bounds():
    q = QualificationResult(icp_score=100)
    assert q.verdict == "needs_review"
    assert q.carrier_fit.confidence == 0.0
