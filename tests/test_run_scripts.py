"""Light tests for the run_cases and analyze_traces scripts."""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_risk_failures():
    run_cases = load_script("run_cases")
    assert run_cases.parse_risk_failures("always") == -1
    assert run_cases.parse_risk_failures("2") == 2
    assert run_cases.parse_risk_failures("0") == 0
    assert run_cases.parse_risk_failures(" 3 ") == 3


def test_fixture_data_shape():
    run_cases = load_script("run_cases")
    entries = run_cases.load_applications(REPO_ROOT / "data" / "applications.json")
    assert [e["case_id"] for e in entries] == [
        "CASE-001",
        "CASE-002",
        "CASE-003",
        "CASE-004",
    ]
    assert [e["expected_decision"] for e in entries] == [
        "approve",
        "deny",
        "refer",
        "approve",
    ]
    assert run_cases.select_cases(entries, "CASE-002")[0]["case_id"] == "CASE-002"
    assert len(run_cases.select_cases(entries, "all")) == 4
    for entry in entries:
        client = run_cases.ScriptedClient(run_cases.SCRIPTED_RESPONSES[entry["case_id"]])
        assert client.responses  # every fixture has a full scripted set


def test_analyze_traces_writes_report_from_fabricated_trace(tmp_path):
    analyzer = load_script("analyze_traces")
    traces = tmp_path / "traces"
    traces.mkdir()
    trace = {
        "trace_id": "t" * 32,
        "case_id": "CASE-TEST",
        "created_at": "2026-10-11T00:00:00+00:00",
        "spans": [
            {
                "trace_id": "t" * 32,
                "span_id": "p" * 32,
                "parent_span_id": None,
                "case_id": "CASE-TEST",
                "agent": "pipeline",
                "attempt": 1,
                "duration_ms": 100.0,
                "usage": {"input_tokens": 0, "output_tokens": 0, "model": ""},
                "status": "ok",
                "error": None,
            },
            {
                "trace_id": "t" * 32,
                "span_id": "a" * 32,
                "parent_span_id": "p" * 32,
                "case_id": "CASE-TEST",
                "agent": "intake_agent",
                "attempt": 1,
                "duration_ms": 40.0,
                "input": {"applicant_name": "Dana Whitfield"},
                "usage": {"input_tokens": 480, "output_tokens": 320, "model": "qwen3:8b"},
                "status": "ok",
                "error": None,
            },
            {
                "trace_id": "t" * 32,
                "span_id": "b" * 32,
                "parent_span_id": "p" * 32,
                "case_id": "CASE-TEST",
                "agent": "risk_scoring_agent",
                "attempt": 1,
                "duration_ms": 60.0,
                "usage": {"input_tokens": 200, "output_tokens": 100, "model": "qwen3:8b"},
                "status": "ok",
                "error": None,
            },
        ],
    }
    (traces / "t.json").write_text(json.dumps(trace), encoding="utf-8")

    out = tmp_path / "docs" / "cost_latency.md"
    code = analyzer.main(["--traces-dir", str(traces), "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "CASE-TEST" in text
    assert "0.00" in text  # actual marginal cost stated as $0.00
    assert "risk_scoring_agent" in text  # slowest agent named from the trace
    assert "https://www.anthropic.com/pricing" in text


def test_run_cases_main_writes_trace_for_single_case(tmp_path):
    run_cases = load_script("run_cases")
    traces = tmp_path / "traces"
    code = run_cases.main(
        [
            "--backend",
            "scripted",
            "--case",
            "CASE-001",
            "--traces-dir",
            str(traces),
        ]
    )
    assert code == 0
    files = list(traces.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    # run_case generates its own case id (uuid hex); the CASE-001 label only
    # selects the fixture, so just check the id shape here.
    assert len(data["case_id"]) == 32
    agents = [s["agent"] for s in data["spans"]]
    assert agents == [
        "pipeline",
        "intake_agent",
        "enrichment_agent",
        "risk_scoring_agent",
        "recommendation_agent",
    ]


def test_analyze_traces_refuses_empty_dir(tmp_path, capsys):
    analyzer = load_script("analyze_traces")
    empty = tmp_path / "empty"
    empty.mkdir()
    code = analyzer.main(["--traces-dir", str(empty), "--out", str(tmp_path / "x.md")])
    assert code == 1
    assert "No traces found" in capsys.readouterr().out
