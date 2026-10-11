"""Pydantic models for the shared case record and agent outputs."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

Product = Literal["auto", "home", "life"]


class RawApplication(BaseModel):
    """The application exactly as submitted, before any normalization."""

    applicant_name: str
    date_of_birth: str | None = None
    address: str | None = None
    product: Product
    coverage_amount: float
    stated_income: float | None = None
    claims_history_text: str | None = None


class IntakeResult(BaseModel):
    """Normalized intake written by intake_agent."""

    applicant_name: str
    date_of_birth: str | None = None
    product: Product | None = None
    coverage_amount: float | None = None
    stated_income: float | None = None
    missing_fields: list[str] = []
    notes: list[str] = []


class EnrichmentResult(BaseModel):
    """Bureau facts plus the model's interpretation, written by enrichment_agent."""

    credit_band: Literal["a", "b", "c", "d", "no_hit"]
    prior_claims_count: int = 0
    asset_age_years: float | None = None
    flags: list[str] = []
    notes: list[str] = []


class RiskFactor(BaseModel):
    """One named contributor to the risk score."""

    name: str
    points: int
    detail: str = ""


class RiskResult(BaseModel):
    """Risk score written by risk_scoring_agent."""

    score: int
    factors: list[RiskFactor] = []
    rationale: str = ""


class RecommendationResult(BaseModel):
    """Final decision written by recommendation_agent."""

    decision: Literal["approve", "deny", "refer"]
    reasons: list[str] = []
    conditions: list[str] = []


CaseStatus = Literal["in_progress", "completed", "escalated"]


class CaseRecord(BaseModel):
    """The shared case record every agent reads from and writes one field to."""

    case_id: str
    created_at: str
    raw: RawApplication
    intake: IntakeResult | None = None
    enrichment: EnrichmentResult | None = None
    risk: RiskResult | None = None
    recommendation: RecommendationResult | None = None
    status: CaseStatus = "in_progress"

    @classmethod
    def new(cls, raw: RawApplication) -> "CaseRecord":
        """Create a fresh record with a generated id and a UTC timestamp."""
        return cls(
            case_id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            raw=raw,
        )
