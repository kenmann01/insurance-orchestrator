"""Tests for the enrichment agent: bureau facts are code-set, the model only interprets."""

from underwriting.agents.enrichment import enrichment_agent
from underwriting.bureau import BUREAU
from underwriting.llm import ScriptedClient
from underwriting.models import IntakeResult


def _client(reply: str) -> ScriptedClient:
    """Scripted client that replies to the enrichment marker with reply."""
    return ScriptedClient({"enrichment agent": reply})


def _intake(name: str) -> IntakeResult:
    """Minimal intake result carrying just the applicant name."""
    return IntakeResult(applicant_name=name, product="auto", coverage_amount=25000.0)


def test_dana_credit_band_comes_from_bureau():
    """The credit band comes from BUREAU; the model's flags ride alongside."""
    reply = '{"flags": ["thin file"], "notes": [], "credit_band": "a"}'
    result = enrichment_agent(_intake("Dana Whitfield"), llm=_client(reply))
    assert result.credit_band == BUREAU["dana whitfield"]["credit_band"] == "a"
    assert result.flags == ["thin file"]
    assert result.prior_claims_count == 0
    assert result.asset_age_years == 3.0


def test_model_cannot_override_bureau_credit_band():
    """A model claiming band a cannot override Marcus Boone's bureau band c."""
    # Marcus Boone is band c in the bureau; the scripted model claims "a".
    reply = '{"flags": [], "notes": [], "credit_band": "a"}'
    result = enrichment_agent(_intake("Marcus Boone"), llm=_client(reply))
    assert result.credit_band == "c"
    assert result.prior_claims_count == 3
    assert result.asset_age_years == 22.0


def test_unknown_applicant_is_no_hit():
    """An applicant absent from BUREAU resolves to the no_hit band with no asset age."""
    # Priya Natarajan is deliberately absent from BUREAU.
    reply = '{"flags": ["no bureau file"], "notes": []}'
    result = enrichment_agent(_intake("Priya Natarajan"), llm=_client(reply))
    assert result.credit_band == "no_hit"
    assert result.prior_claims_count == 0
    assert result.asset_age_years is None
