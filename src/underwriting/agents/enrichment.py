"""Enrichment agent: interpret deterministic bureau facts.

Bureau facts (credit band, prior claims, asset age) come from code, never
from the model. The model only produces flags and notes that interpret those
facts for a human reviewer, and any credit fields it emits are ignored.
"""

import json
from typing import Any

from pydantic import BaseModel

from underwriting import bureau as bureau_module
from underwriting.llm import LLMClient, parse_json_model
from underwriting.models import EnrichmentResult, IntakeResult

ENRICHMENT_SYSTEM = (
    "You are the enrichment agent. You interpret bureau facts for an "
    "underwriting case.\n"
    "You receive the applicant intake JSON and a bureau facts JSON produced by "
    "deterministic code.\n"
    'Return ONLY a JSON object of the form {"flags": [...], "notes": [...]} '
    "where both values are lists of strings.\n"
    "Interpret the facts for a human reviewer: for example an old asset, prior "
    "claims, or a weak credit band belong in flags or notes.\n"
    "Never output credit data fields in your reply: do not include credit_band, "
    "prior_claims_count, or asset_age_years. Those fields are set by code, "
    "never by the model."
)


class EnrichmentInterpretation(BaseModel):
    """The only part of enrichment the model is allowed to produce."""

    flags: list[str] = []
    notes: list[str] = []


def enrichment_agent(
    intake: IntakeResult,
    *,
    llm: LLMClient,
    bureau: Any = bureau_module,
) -> EnrichmentResult:
    """Look up bureau facts by name and let the model interpret them.

    The bureau parameter defaults to the bureau module (which provides
    lookup()); tests can inject any object with a matching lookup method.
    Credit fields are copied from the facts dict only; the model's reply is
    parsed into EnrichmentInterpretation, which has no credit fields at all.
    """
    facts = bureau.lookup(intake.applicant_name)
    user = (
        "Applicant intake JSON:\n"
        + intake.model_dump_json()
        + "\n\nBureau facts JSON:\n"
        + json.dumps(facts)
    )
    response = llm.complete(ENRICHMENT_SYSTEM, user)
    parsed = parse_json_model(
        llm, ENRICHMENT_SYSTEM, user, response.text, EnrichmentInterpretation
    )
    return EnrichmentResult(
        credit_band=facts["credit_band"],
        prior_claims_count=facts.get("prior_claims_count", 0),
        asset_age_years=facts.get("asset_age_years"),
        flags=parsed.flags,
        notes=parsed.notes,
    )
