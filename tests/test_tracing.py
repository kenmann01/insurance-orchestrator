"""Tests for the span tracer and the usage recording client."""

import json
import time

import pytest

from underwriting.llm import LLMResponse, ScriptedClient, UsageRecordingClient
from underwriting.tracing import Tracer, record_usage, set_usage


def test_span_records_duration_and_ok_status():
    tracer = Tracer("case-1")
    with tracer.pipeline_span():
        with tracer.span("intake_agent") as span:
            time.sleep(0.005)
            span.input = {"x": 1}
    assert span.status == "ok"
    assert span.error is None
    assert span.duration_ms > 0
    assert span.ended_at is not None
    assert span.started_at <= span.ended_at


def test_exception_marks_error_and_reraises():
    tracer = Tracer("case-1")
    with pytest.raises(ValueError, match="boom"):
        with tracer.pipeline_span():
            with tracer.span("risk_scoring_agent", attempt=2) as span:
                raise ValueError("boom")
    assert len(tracer.spans) == 2
    assert tracer.spans[-1].status == "error"
    assert tracer.spans[-1].error == "ValueError: boom"
    assert tracer.spans[-1].duration_ms >= 0


def test_dump_writes_parent_child_structure(tmp_path):
    tracer = Tracer("case-42")
    with tracer.pipeline_span() as pipeline:
        with tracer.span("intake_agent", input={"a": 1}) as intake_span:
            intake_span.output = {"b": 2}
        with tracer.span("enrichment_agent"):
            pass
    path = tmp_path / "trace.json"
    tracer.dump(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["trace_id"] == tracer.trace_id
    assert data["case_id"] == "case-42"
    assert data["created_at"] == tracer.created_at
    agents = [s["agent"] for s in data["spans"]]
    assert agents == ["pipeline", "intake_agent", "enrichment_agent"]
    pipeline_dict = data["spans"][0]
    assert pipeline_dict["parent_span_id"] is None
    for child in data["spans"][1:]:
        assert child["parent_span_id"] == pipeline_dict["span_id"]
    assert data["spans"][1]["input"] == {"a": 1}
    assert data["spans"][1]["output"] == {"b": 2}
    assert pipeline.span_id == pipeline_dict["span_id"]


def test_summary_contains_agents_and_total_row():
    tracer = Tracer("case-7")
    with tracer.pipeline_span():
        with tracer.span("intake_agent") as span:
            set_usage(span, 10, 5, "scripted")
        with tracer.span("risk_scoring_agent", attempt=2):
            pass
    text = tracer.summary()
    assert "intake_agent" in text
    assert "risk_scoring_agent" in text
    assert "pipeline" in text
    assert "TOTAL" in text
    total_line = next(line for line in text.splitlines() if line.startswith("TOTAL"))
    assert "10" in total_line  # summed tokens in
    assert "5" in total_line  # summed tokens out


def test_set_usage_and_record_usage():
    tracer = Tracer("case-1")
    with tracer.span("intake_agent") as span:
        set_usage(span, 3, 4, "m1")
    assert span.usage == {"input_tokens": 3, "output_tokens": 4, "model": "m1"}

    responses = [
        LLMResponse(text="a", input_tokens=7, output_tokens=2, model="m2"),
        LLMResponse(text="b", input_tokens=1, output_tokens=3, model="m3"),
    ]
    with tracer.span("enrichment_agent") as span2:
        record_usage(span2, responses)
    assert span2.usage == {"input_tokens": 8, "output_tokens": 5, "model": "m3"}


def test_usage_recording_client_drains_and_clears():
    inner = ScriptedClient({"intake agent": '{"x": 1}'})
    recorder = UsageRecordingClient(inner)
    assert recorder.complete("You are the intake agent", "hi").text == '{"x": 1}'
    assert recorder.complete("You are the intake agent", "again").text == '{"x": 1}'
    drained = recorder.drain()
    assert len(drained) == 2
    assert drained[0].input_tokens == 120
    assert recorder.drain() == []
    recorder.complete("You are the intake agent", "more")
    recorder.clear()
    assert recorder.records == []
