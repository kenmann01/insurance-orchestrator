"""Failure drills: retry-then-escalate behavior of the risk scoring step."""

import json

from underwriting.llm import ScriptedClient, TimeoutRiskClient
from underwriting.models import RawApplication
from underwriting.pipeline import run_case

MARCUS = RawApplication(
    applicant_name="Marcus Boone",
    date_of_birth="1978-11-02",
    address="9 Granite Court, Springfield",
    product="home",
    coverage_amount=1200000.0,
    stated_income=48000.0,
    claims_history_text=(
        "Three prior claims in five years: water damage, fire, and theft. "
        "Home is 22 years old."
    ),
)

# Score 70 with factors summing to 70: the gate denies at 61-100.
SCRIPTED_RESPONSES = {
    "intake agent": json.dumps(
        {
            "applicant_name": "Marcus Boone",
            "date_of_birth": "1978-11-02",
            "product": "home",
            "coverage_amount": 1200000,
            "stated_income": 48000,
            "missing_fields": [],
            "notes": [],
        }
    ),
    "enrichment agent": json.dumps(
        {
            "flags": ["three prior claims in five years", "home is 22 years old"],
            "notes": ["credit band c"],
        }
    ),
    "risk scoring agent": json.dumps(
        {
            "score": 70,
            "factors": [
                {"name": "weak credit", "points": 35, "detail": "band c"},
                {"name": "prior claims", "points": 25, "detail": "three in five years"},
                {"name": "aging asset", "points": 10, "detail": "22 year old home"},
            ],
            "rationale": "high risk home case",
        }
    ),
    "recommendation agent": json.dumps(
        {
            "decision": "deny",
            "reasons": ["score 70 is within the deny band"],
            "conditions": [],
        }
    ),
}


def read_trace(tmp_path) -> dict:
    return json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))


def test_always_failing_risk_escalates_to_refer(tmp_path):
    client = TimeoutRiskClient(ScriptedClient(SCRIPTED_RESPONSES), fail_first_n=-1)
    case = run_case(MARCUS, client, traces_dir=tmp_path)

    # run_case returns normally: escalation is a handled outcome, not a crash.
    assert case.status == "escalated"
    assert case.risk is None
    assert case.recommendation is not None
    assert case.recommendation.decision == "refer"
    assert "risk scoring unavailable" in case.recommendation.reasons[0]
    assert case.recommendation.conditions == []

    data = read_trace(tmp_path)
    risk_spans = [s for s in data["spans"] if s["agent"] == "risk_scoring_agent"]
    assert [s["attempt"] for s in risk_spans] == [1, 2, 3]
    for span in risk_spans:
        assert span["status"] in {"retry", "error"}
        assert span["error"] == "TimeoutError: simulated risk scoring timeout"
    assert risk_spans[-1]["status"] == "error"
    assert risk_spans[0]["status"] == "retry"
    assert risk_spans[1]["status"] == "retry"
    assert any(s["status"] == "escalated" for s in data["spans"])

    # No decision was invented anywhere in the trace.
    for span in data["spans"]:
        text = json.dumps(span["output"])
        assert "approve" not in text
        assert "deny" not in text

    agents = [s["agent"] for s in data["spans"]]
    assert agents == [
        "pipeline",
        "intake_agent",
        "enrichment_agent",
        "risk_scoring_agent",
        "risk_scoring_agent",
        "risk_scoring_agent",
        "pipeline_escalation",
    ]


def test_one_risk_failure_recovers_on_second_attempt(tmp_path):
    client = TimeoutRiskClient(ScriptedClient(SCRIPTED_RESPONSES), fail_first_n=1)
    case = run_case(MARCUS, client, traces_dir=tmp_path)

    assert case.status == "completed"
    assert case.risk is not None and case.risk.score == 70
    assert case.recommendation is not None
    assert case.recommendation.decision == "deny"

    data = read_trace(tmp_path)
    risk_spans = [s for s in data["spans"] if s["agent"] == "risk_scoring_agent"]
    assert len(risk_spans) == 2
    assert risk_spans[0]["attempt"] == 1
    assert risk_spans[0]["status"] == "retry"
    assert risk_spans[0]["error"] == "TimeoutError: simulated risk scoring timeout"
    assert risk_spans[1]["attempt"] == 2
    assert risk_spans[1]["status"] == "ok"
    assert risk_spans[1]["error"] is None
    assert risk_spans[1]["output"]["score"] == 70
    # The successful retry attaches real usage from the scripted client.
    assert risk_spans[1]["usage"]["model"] == "scripted"
    assert risk_spans[1]["usage"]["input_tokens"] > 0


def test_timeout_risk_client_passes_non_risk_calls_through():
    inner = ScriptedClient(SCRIPTED_RESPONSES)
    client = TimeoutRiskClient(inner, fail_first_n=1)

    intake = client.complete("You are the intake agent", "Application JSON: {}")
    assert json.loads(intake.text)["applicant_name"] == "Marcus Boone"

    import pytest

    with pytest.raises(TimeoutError, match="simulated risk scoring timeout"):
        client.complete("You are the risk scoring agent", "Intake JSON: {}")

    # Second risk call goes through: the fail budget was spent.
    risk = client.complete("You are the risk scoring agent", "Intake JSON: {}")
    assert json.loads(risk.text)["score"] == 70
