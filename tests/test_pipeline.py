"""Tests for the four-agent pipeline on the shared case record."""

import json

import pytest

from underwriting.llm import ScriptedClient
from underwriting.models import RawApplication
from underwriting.pipeline import RetryableError, call_with_retry, run_case

DANA = RawApplication(
    applicant_name="Dana Whitfield",
    date_of_birth="1990-04-12",
    address="48 Alder Lane, Springfield",
    product="auto",
    coverage_amount=42000.0,
    stated_income=78000.0,
    claims_history_text="No prior claims. Vehicle is 3 years old.",
)

SCRIPTED_RESPONSES = {
    "intake agent": json.dumps(
        {
            "applicant_name": "Dana Whitfield",
            "date_of_birth": "1990-04-12",
            "product": "auto",
            "coverage_amount": 42000,
            "stated_income": 78000,
            "missing_fields": [],
            "notes": [],
        }
    ),
    "enrichment agent": json.dumps(
        {
            "flags": ["asset is 3 years old"],
            "notes": ["credit band a, no prior claims on file"],
        }
    ),
    # Score 18 with factors summing to 18: the gate approves at 0-30.
    "risk scoring agent": json.dumps(
        {
            "score": 18,
            "factors": [
                {"name": "clean credit", "points": 8, "detail": "band a"},
                {"name": "no prior claims", "points": 10, "detail": "none on file"},
            ],
            "rationale": "low risk auto case",
        }
    ),
    "recommendation agent": json.dumps(
        {
            "decision": "approve",
            "reasons": ["score 18 is within the approve band"],
            "conditions": [],
        }
    ),
}


def test_call_with_retry_success_first_try():
    calls = []
    result, used = call_with_retry(lambda: calls.append(1) or "ok", sleep=lambda s: None)
    assert result == "ok"
    assert used == [1]


def test_call_with_retry_retries_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("transient")
        return "done"

    result, used = call_with_retry(flaky, sleep=sleeps.append)
    assert result == "done"
    assert used == [1, 2, 3]
    assert sleeps == [0.05, 0.1]


def test_call_with_retry_exhausts_and_raises():
    def always_fails():
        raise RetryableError("down")

    with pytest.raises(RetryableError, match="down"):
        call_with_retry(always_fails, attempts=3, sleep=lambda s: None)


def test_run_case_completes_with_approve(tmp_path):
    client = ScriptedClient(SCRIPTED_RESPONSES)
    case = run_case(DANA, client, traces_dir=tmp_path)

    assert case.status == "completed"
    assert case.case_id
    assert case.intake is not None and case.intake.applicant_name == "Dana Whitfield"
    assert case.enrichment is not None and case.enrichment.credit_band == "a"
    assert case.risk is not None and case.risk.score == 18
    assert case.recommendation is not None
    assert case.recommendation.decision == "approve"


def test_run_case_trace_structure_and_handoff(tmp_path):
    client = ScriptedClient(SCRIPTED_RESPONSES)
    case = run_case(DANA, client, traces_dir=tmp_path)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["case_id"] == case.case_id

    agents = [s["agent"] for s in data["spans"]]
    assert agents == [
        "pipeline",
        "intake_agent",
        "enrichment_agent",
        "risk_scoring_agent",
        "recommendation_agent",
    ]
    statuses = [s["status"] for s in data["spans"]]
    assert statuses == ["ok", "ok", "ok", "ok", "ok"]
    assert all(s["error"] is None for s in data["spans"])

    pipeline_span = data["spans"][0]
    assert pipeline_span["parent_span_id"] is None
    for child in data["spans"][1:]:
        assert child["parent_span_id"] == pipeline_span["span_id"]

    # HANDOFF: the enrichment span input carries the intake output values.
    intake_span = data["spans"][1]
    enrichment_span = data["spans"][2]
    assert (
        enrichment_span["input"]["applicant_name"]
        == intake_span["output"]["applicant_name"]
    )
    assert enrichment_span["input"]["product"] == intake_span["output"]["product"]
    assert intake_span["output"]["coverage_amount"] == 42000


def test_run_case_span_usage_attached_per_agent(tmp_path):
    client = ScriptedClient(SCRIPTED_RESPONSES)
    run_case(DANA, client, traces_dir=tmp_path)

    data = json.loads(
        next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    )
    agent_spans = [s for s in data["spans"] if s["agent"] != "pipeline"]
    assert len(agent_spans) == 4
    for span in agent_spans:
        usage = span["usage"]
        assert usage["model"] == "scripted"
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
