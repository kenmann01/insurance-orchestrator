from underwriting.agents.recommendation import gate_decision, recommendation_agent
from underwriting.llm import ScriptedClient
from underwriting.models import EnrichmentResult, IntakeResult, RiskResult


def _client(reply: str) -> ScriptedClient:
    return ScriptedClient({"recommendation agent": reply})


def _clean() -> tuple[IntakeResult, EnrichmentResult, RiskResult]:
    intake = IntakeResult(
        applicant_name="Dana Whitfield",
        date_of_birth="1990-04-12",
        product="auto",
        coverage_amount=25000.0,
    )
    enrichment = EnrichmentResult(
        credit_band="a", prior_claims_count=0, asset_age_years=3.0
    )
    return intake, enrichment, RiskResult(score=18, rationale="clean file")


def test_recommendation_approve_case():
    intake, enrichment, risk = _clean()
    reply = '{"decision": "approve", "reasons": ["clean file"], "conditions": []}'
    result = recommendation_agent(intake, enrichment, risk, llm=_client(reply))
    assert result.decision == "approve"
    assert result.reasons == ["clean file"]


def test_gate_overrides_contradicting_model():
    intake, enrichment, risk = _clean()
    risk.score = 85  # gate says deny
    reply = '{"decision": "approve", "reasons": ["model disagrees"], "conditions": []}'
    result = recommendation_agent(intake, enrichment, risk, llm=_client(reply))
    assert result.decision == "deny"
    assert any(r.startswith("gate held") for r in result.reasons)


def test_no_hit_band_forces_refer():
    intake, enrichment, risk = _clean()
    enrichment = enrichment.model_copy(update={"credit_band": "no_hit"})
    reply = '{"decision": "approve", "reasons": [], "conditions": []}'
    result = recommendation_agent(intake, enrichment, risk, llm=_client(reply))
    assert result.decision == "refer"


def test_gate_decision_rules_directly():
    intake, enrichment, risk = _clean()
    assert gate_decision(intake, enrichment, risk) == "approve"
    assert gate_decision(intake, enrichment, RiskResult(score=45)) == "refer"
    assert gate_decision(intake, enrichment, RiskResult(score=70)) == "deny"
    assert gate_decision(intake, enrichment, None) == "refer"

    missing_dob = intake.model_copy(update={"missing_fields": ["date_of_birth"]})
    assert gate_decision(missing_dob, enrichment, risk) == "refer"
