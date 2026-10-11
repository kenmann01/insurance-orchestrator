"""Tests for the risk scoring agent's clamp and factor-sum validation."""

import pytest

from underwriting.agents.risk import risk_scoring_agent
from underwriting.llm import AgentOutputError, ScriptedClient
from underwriting.models import EnrichmentResult, IntakeResult


def _client(reply: str) -> ScriptedClient:
    """Scripted client that replies to the risk scoring marker with reply."""
    return ScriptedClient({"risk scoring agent": reply})


def _inputs() -> tuple[IntakeResult, EnrichmentResult]:
    """A clean intake and enrichment pair ready to be scored."""
    intake = IntakeResult(
        applicant_name="Dana Whitfield",
        date_of_birth="1990-04-12",
        product="auto",
        coverage_amount=25000.0,
    )
    enrichment = EnrichmentResult(
        credit_band="a", prior_claims_count=0, asset_age_years=3.0
    )
    return intake, enrichment


def test_risk_happy_path():
    """A score whose factor points add up exactly passes validation unchanged."""
    intake, enrichment = _inputs()
    reply = (
        '{"score": 22, "factors": [{"name": "claims", "points": 12, '
        '"detail": "minor history"}, {"name": "asset age", "points": 10, '
        '"detail": "newer vehicle"}], "rationale": "moderate risk"}'
    )
    result = risk_scoring_agent(intake, enrichment, llm=_client(reply))
    assert result.score == 22
    assert len(result.factors) == 2
    assert result.rationale == "moderate risk"


def test_risk_score_is_clamped_into_range():
    """An out-of-range score is clamped to 100 while the factor sum still matches it."""
    intake, enrichment = _inputs()
    reply = (
        '{"score": 130, "factors": [{"name": "claims", "points": 60, '
        '"detail": ""}, {"name": "credit band", "points": 40, "detail": ""}], '
        '"rationale": "severe"}'
    )
    result = risk_scoring_agent(intake, enrichment, llm=_client(reply))
    assert result.score == 100
    assert sum(f.points for f in result.factors) == 100


def test_risk_factor_sum_mismatch_raises():
    """Factor points that miss the score raise AgentOutputError."""
    intake, enrichment = _inputs()
    reply = (
        '{"score": 40, "factors": [{"name": "claims", "points": 90, '
        '"detail": ""}], "rationale": "mismatched factors"}'
    )
    with pytest.raises(AgentOutputError, match="does not match score"):
        risk_scoring_agent(intake, enrichment, llm=_client(reply))
