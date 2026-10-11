"""Risk scoring agent with factor-sum validation.

The model proposes a 0-100 score with a factor breakdown. Code clamps the
score into range and rejects the output when the factor points do not
actually add up to the score (within a small tolerance), because an
unexplained score is exactly the kind of output an underwriter cannot trust.
"""

from underwriting.llm import AgentOutputError, LLMClient, complete_parsed
from underwriting.models import EnrichmentResult, IntakeResult, RiskResult

RISK_SYSTEM = (
    "You are the risk scoring agent. You score the underwriting risk of a "
    "case.\n"
    "You receive intake JSON and enrichment JSON.\n"
    'Return ONLY a JSON object of the form {"score": <int 0-100, higher is '
    'worse>, "factors": [{"name": ..., "points": <int>, "detail": ...}], '
    '"rationale": "..."}.\n'
    "Factor points MUST sum to the score."
)

SCORE_MIN = 0
SCORE_MAX = 100
FACTOR_SUM_TOLERANCE = 5


def risk_scoring_agent(
    intake: IntakeResult,
    enrichment: EnrichmentResult,
    *,
    llm: LLMClient,
) -> RiskResult:
    user = (
        "Intake JSON:\n"
        + intake.model_dump_json()
        + "\n\nEnrichment JSON:\n"
        + enrichment.model_dump_json()
    )
    result = complete_parsed(llm, RISK_SYSTEM, user, RiskResult)

    score = max(SCORE_MIN, min(SCORE_MAX, int(result.score)))
    factors = list(result.factors) if result.factors else []
    total = sum(f.points for f in factors)
    if abs(total - score) > FACTOR_SUM_TOLERANCE:
        raise AgentOutputError(
            f"factor points sum {total} does not match score {score} "
            f"within tolerance {FACTOR_SUM_TOLERANCE}"
        )
    result.score = score
    result.factors = factors
    return result
