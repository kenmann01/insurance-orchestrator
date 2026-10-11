"""Recommendation agent with a deterministic policy gate.

The policy is code, not prompt: gate_decision() computes the decision from
the case sections, the model is asked to apply the same rules, and code
overrides any contradiction. A model can add color (reasons, conditions) but
it cannot move the decision.
"""

from typing import Literal

from underwriting.llm import LLMClient, parse_json_model
from underwriting.models import (
    EnrichmentResult,
    IntakeResult,
    RecommendationResult,
    RiskResult,
)

RECOMMENDATION_SYSTEM = (
    "You are the recommendation agent. You issue the final underwriting "
    "decision for a case.\n"
    "You receive intake JSON, enrichment JSON, risk JSON, and the current "
    "policy gate evaluation.\n"
    'Return ONLY a JSON object of the form {"decision": "approve" or "deny" '
    'or "refer", "reasons": [...], "conditions": [...]}, where reasons and '
    "conditions are lists of strings.\n"
    "Policy rules:\n"
    "- A missing critical field (date_of_birth, product, coverage_amount) "
    "forces refer.\n"
    "- credit_band no_hit forces refer.\n"
    "- Score 0-30 approves.\n"
    "- Score 31-60 refers.\n"
    "- Score 61-100 denies.\n"
    "Apply the rules yourself; the gate line in the user message is a "
    "cross-check, not a value to copy blindly."
)

Decision = Literal["approve", "deny", "refer"]

_CRITICAL_FIELDS = ("date_of_birth", "product", "coverage_amount")


def gate_decision(
    intake: IntakeResult,
    enrichment: EnrichmentResult,
    risk: RiskResult | None,
) -> Decision:
    """Deterministic policy decision computed from the case sections."""
    for field in _CRITICAL_FIELDS:
        if field in intake.missing_fields:
            return "refer"
    if enrichment.credit_band == "no_hit":
        return "refer"
    if risk is None:
        return "refer"
    if risk.score <= 30:
        return "approve"
    if risk.score <= 60:
        return "refer"
    return "deny"


def recommendation_agent(
    intake: IntakeResult,
    enrichment: EnrichmentResult,
    risk: RiskResult | None,
    *,
    llm: LLMClient,
) -> RecommendationResult:
    expected = gate_decision(intake, enrichment, risk)
    user = (
        "Intake JSON:\n"
        + intake.model_dump_json()
        + "\n\nEnrichment JSON:\n"
        + enrichment.model_dump_json()
        + "\n\nRisk JSON:\n"
        + (risk.model_dump_json() if risk is not None else "null")
        + f"\n\nThe policy gate currently evaluates this case as: {expected}. "
        "Apply the rules yourself and state your decision."
    )
    response = llm.complete(RECOMMENDATION_SYSTEM, user)
    parsed = parse_json_model(
        llm, RECOMMENDATION_SYSTEM, user, response.text, RecommendationResult
    )
    if parsed.decision != expected:
        parsed.decision = expected
        parsed.reasons.append("gate held: policy decision is " + expected)
    return parsed
