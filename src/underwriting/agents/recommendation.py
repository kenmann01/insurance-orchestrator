"""Recommendation agent with a deterministic policy gate.

The policy is code, not prompt: gate_decision() computes the decision from
the case sections, the model is asked to apply the same rules, and code
overrides any contradiction. A model can add color (reasons, conditions) but
it cannot move the decision.
"""

from typing import Literal

from underwriting.llm import LLMClient, complete_parsed
from underwriting.models import (
    EnrichmentResult,
    IntakeResult,
    RecommendationResult,
    RiskResult,
)

APPROVE_MAX_SCORE = 30
REFER_MAX_SCORE = 60

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
    f"- Score 0-{APPROVE_MAX_SCORE} approves.\n"
    f"- Score {APPROVE_MAX_SCORE + 1}-{REFER_MAX_SCORE} refers.\n"
    f"- Score {REFER_MAX_SCORE + 1}-100 denies.\n"
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
    if risk.score <= APPROVE_MAX_SCORE:
        return "approve"
    if risk.score <= REFER_MAX_SCORE:
        return "refer"
    return "deny"


def recommendation_agent(
    intake: IntakeResult,
    enrichment: EnrichmentResult,
    risk: RiskResult | None,
    *,
    llm: LLMClient,
) -> RecommendationResult:
    """Produce the final RecommendationResult under the policy gate.

    The model drafts decision, reasons, and conditions; code first computes
    gate_decision() from the case sections and overrides any contradicting
    model decision (appending a "gate held" reason), so the model can add
    color but cannot move the decision.
    """
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
    parsed = complete_parsed(llm, RECOMMENDATION_SYSTEM, user, RecommendationResult)
    if parsed.decision != expected:
        parsed.decision = expected
        parsed.reasons.append("gate held: policy decision is " + expected)
    return parsed
