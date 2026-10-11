from underwriting.agents.enrichment import enrichment_agent
from underwriting.bureau import BUREAU
from underwriting.llm import ScriptedClient
from underwriting.models import IntakeResult


def _client(reply: str) -> ScriptedClient:
    return ScriptedClient({"enrichment agent": reply})


def _intake(name: str) -> IntakeResult:
    return IntakeResult(applicant_name=name, product="auto", coverage_amount=25000.0)


def test_dana_credit_band_comes_from_bureau():
    reply = '{"flags": ["thin file"], "notes": [], "credit_band": "a"}'
    result = enrichment_agent(_intake("Dana Whitfield"), llm=_client(reply))
    assert result.credit_band == BUREAU["dana whitfield"]["credit_band"] == "a"
    assert result.flags == ["thin file"]
    assert result.prior_claims_count == 0
    assert result.asset_age_years == 3.0


def test_model_cannot_override_bureau_credit_band():
    # Marcus Boone is band c in the bureau; the scripted model claims "a".
    reply = '{"flags": [], "notes": [], "credit_band": "a"}'
    result = enrichment_agent(_intake("Marcus Boone"), llm=_client(reply))
    assert result.credit_band == "c"
    assert result.prior_claims_count == 3
    assert result.asset_age_years == 22.0


def test_unknown_applicant_is_no_hit():
    # Priya Natarajan is deliberately absent from BUREAU.
    reply = '{"flags": ["no bureau file"], "notes": []}'
    result = enrichment_agent(_intake("Priya Natarajan"), llm=_client(reply))
    assert result.credit_band == "no_hit"
    assert result.prior_claims_count == 0
    assert result.asset_age_years is None
