"""Deterministic stand-in for an external credit-bureau API.

The entries below are fixed lookup data set by code. The model never sets
credit data: enrichment_agent reads bureau facts from here and only interprets
them. Names are lowercase full names.
"""

BUREAU: dict[str, dict] = {
    "dana whitfield": {"credit_band": "a", "prior_claims_count": 0, "asset_age_years": 3.0},
    "marcus boone": {"credit_band": "c", "prior_claims_count": 3, "asset_age_years": 22.0},
    "elena petrov": {"credit_band": "b", "prior_claims_count": 1, "asset_age_years": 6.0},
}


def lookup(name: str) -> dict:
    """Return a copy of the bureau entry for name, or a no_hit marker.

    A missing applicant (for example priya natarajan) yields credit_band
    "no_hit" instead of raising, so the case can be referred, not crashed.
    """
    key = name.strip().lower()
    entry = BUREAU.get(key)
    if entry is None:
        return {"credit_band": "no_hit"}
    return dict(entry)
