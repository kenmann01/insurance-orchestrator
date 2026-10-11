"""The four-agent pipeline on one shared case record, with tracing and retries.

run_case() walks the fixed sequence intake, enrichment, risk scoring,
recommendation. Every agent call runs inside a tracer span, all LLM traffic
goes through one shared UsageRecordingClient so each span can attach the
usage of exactly its own calls, and risk scoring retries on transient
failures before the case is escalated to a deterministic referral.
"""

import time
from pathlib import Path
from typing import Callable, TypeVar

from underwriting.agents.intake import intake_agent
from underwriting.agents.enrichment import enrichment_agent
from underwriting.agents.risk import risk_scoring_agent
from underwriting.agents.recommendation import recommendation_agent
from underwriting.llm import LLMClient, UsageRecordingClient
from underwriting.models import (
    CaseRecord,
    RawApplication,
    RecommendationResult,
    RiskResult,
)
from underwriting.tracing import Tracer, record_usage


class RetryableError(Exception):
    """An agent failure that is worth another attempt (transport level)."""


RETRYABLE = (RetryableError, TimeoutError)

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.05, 0.1)

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, list[int]]:
    """Call fn(), retrying retryable failures with backoff.

    Tries up to attempts times. After a retryable failure that still leaves
    budget, sleeps backoff[min(attempt - 1, len(backoff) - 1)] seconds, so
    the last backoff value repeats once the list runs out. Returns
    (result, attempts_used) where attempts_used lists every attempt number
    tried, 1-indexed. Raises the last failure after exhausting the budget.
    """
    attempts_used: list[int] = []
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        attempts_used.append(attempt)
        try:
            return fn(), attempts_used
        except RETRYABLE as exc:
            last = exc
            if attempt < attempts:
                sleep(backoff[min(attempt - 1, len(backoff) - 1)])
    if last is None:
        raise AssertionError("retry loop ended without a result or a failure")
    raise last


def run_case(
    raw: RawApplication,
    llm: LLMClient,
    *,
    traces_dir: Path | None = None,
    max_attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> CaseRecord:
    """Run the full pipeline for one application and return the case record.

    When traces_dir is given the tracer dump is always written, escalated or
    not, as traces_dir / <trace_id>.json.
    """
    case = CaseRecord.new(raw)
    tracer = Tracer(case.case_id)
    recorder = UsageRecordingClient(llm)

    with tracer.pipeline_span():
        with tracer.span("intake_agent", input=raw.model_dump(mode="json")) as span:
            intake = intake_agent(raw, llm=recorder)
            span.output = intake.model_dump(mode="json")
            record_usage(span, recorder.drain())
            case.intake = intake

        with tracer.span(
            "enrichment_agent", input=intake.model_dump(mode="json")
        ) as span:
            enrichment = enrichment_agent(intake, llm=recorder)
            span.output = enrichment.model_dump(mode="json")
            record_usage(span, recorder.drain())
            case.enrichment = enrichment

        risk_input = {
            "intake": intake.model_dump(mode="json"),
            "enrichment": enrichment.model_dump(mode="json"),
        }
        attempts_made = 0

        def score_risk() -> RiskResult:
            """One traced risk scoring attempt; marks the span on failure."""
            nonlocal attempts_made
            attempts_made += 1
            try:
                with tracer.span(
                    "risk_scoring_agent", attempt=attempts_made, input=risk_input
                ) as span:
                    scored = risk_scoring_agent(intake, enrichment, llm=recorder)
                    span.output = scored.model_dump(mode="json")
                    record_usage(span, recorder.drain())
                    return scored
            except RETRYABLE as exc:
                # The failure propagated through the span context manager, so
                # the span already closed as an error; re-mark it retry or
                # error by remaining budget, then re-raise for call_with_retry.
                span.status = (
                    "retry" if attempts_made < max_attempts else "error"
                )
                span.error = f"{type(exc).__name__}: {exc}"
                raise

        try:
            # The tested retry path: call_with_retry owns attempt numbering,
            # the backoff schedule, and raising the last failure once the
            # budget is spent.
            risk, _attempts_used = call_with_retry(
                score_risk, attempts=max_attempts, backoff=backoff, sleep=sleep
            )
        except RETRYABLE:
            risk = None
        case.risk = risk

        if risk is None:
            # Escalated path: no risk score exists, so no LLM decision is
            # invented. Code refers the case for a human with a fixed reason.
            referral = RecommendationResult(
                decision="refer",
                reasons=[f"risk scoring unavailable after {max_attempts} attempts"],
                conditions=[],
            )
            case.status = "escalated"
            case.recommendation = referral
            with tracer.span("pipeline_escalation") as span:
                span.output = referral.model_dump(mode="json")
                span.status = "escalated"
        else:
            recommendation_input = {
                "intake": intake.model_dump(mode="json"),
                "enrichment": enrichment.model_dump(mode="json"),
                "risk": risk.model_dump(mode="json"),
            }
            with tracer.span(
                "recommendation_agent", input=recommendation_input
            ) as span:
                recommendation = recommendation_agent(
                    intake, enrichment, risk, llm=recorder
                )
                span.output = recommendation.model_dump(mode="json")
                record_usage(span, recorder.drain())
                case.recommendation = recommendation
                case.status = "completed"

    if traces_dir is not None:
        tracer.dump(Path(traces_dir) / (tracer.trace_id + ".json"))
    return case
