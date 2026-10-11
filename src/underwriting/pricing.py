"""Cost model: hosted-equivalent USD for the token usage recorded in traces.

HOSTED_RATES_PER_MTOK are hosted-equivalent rates for a Sonnet-class model,
from https://www.anthropic.com/pricing, rates as of the build date (2026-10).
Inference in this lab runs locally, so the actual marginal cost is $0.00;
the hosted-equivalent number answers what the same token volume would have
cost on a hosted API.
"""

HOSTED_RATES_PER_MTOK = {"input": 3.0, "output": 15.0}

_TOKENS_PER_MTOK = 1_000_000.0


def span_cost(input_tokens: int, output_tokens: int) -> float:
    """Hosted-equivalent USD for one span's token totals."""
    input_usd = (input_tokens / _TOKENS_PER_MTOK) * HOSTED_RATES_PER_MTOK["input"]
    output_usd = (output_tokens / _TOKENS_PER_MTOK) * HOSTED_RATES_PER_MTOK["output"]
    return input_usd + output_usd


def case_cost(spans: list[dict]) -> float:
    """Hosted-equivalent USD summed over trace-shaped span dicts."""
    total = 0.0
    for span in spans:
        usage = span.get("usage") or {}
        total += span_cost(
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )
    return total
