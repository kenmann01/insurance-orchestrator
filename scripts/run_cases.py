"""Run fixture underwriting cases through the four-agent pipeline.

Examples:
    uv run scripts/run_cases.py                          # all cases on local Ollama
    uv run scripts/run_cases.py --backend scripted       # offline, deterministic
    uv run scripts/run_cases.py --case CASE-004 --risk-failures always
"""

import argparse
import json
import sys
from pathlib import Path

from underwriting.llm import OllamaClient, ScriptedClient, TimeoutRiskClient
from underwriting.models import RawApplication
from underwriting.pipeline import run_case
from underwriting.tracing import Tracer

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_risk_failures(value: str) -> int:
    """Parse the --risk-failures argument: 'always' means -1, else an int."""
    if value.strip().lower() == "always":
        return -1
    return int(value)


def load_applications(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_cases(entries: list[dict], which: str) -> list[dict]:
    """Return all entries, or the one entry matching a case id."""
    if which == "all":
        return list(entries)
    matches = [entry for entry in entries if entry["case_id"] == which]
    if not matches:
        known = ", ".join(entry["case_id"] for entry in entries)
        raise SystemExit(f"unknown case id {which!r}; known ids: {known}")
    return matches


# Scripted backend replies, one set per fixture case. Risk factor points sum
# exactly to the score and each scripted decision matches what the policy
# gate computes for that case (CASE-003 refers because the bureau has no
# record for Priya Natarajan, so the gate refers regardless of the score).
SCRIPTED_RESPONSES: dict[str, dict[str, str]] = {
    "CASE-001": {
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
                "flags": [],
                "notes": ["credit band a", "no prior claims", "vehicle is 3 years old"],
            }
        ),
        "risk scoring agent": json.dumps(
            {
                "score": 15,
                "factors": [
                    {"name": "clean credit", "points": 8, "detail": "credit band a"},
                    {"name": "no prior claims", "points": 7, "detail": "none on file"},
                ],
                "rationale": "low risk auto case",
            }
        ),
        "recommendation agent": json.dumps(
            {
                "decision": "approve",
                "reasons": ["score 15 is within the approve band"],
                "conditions": [],
            }
        ),
    },
    "CASE-002": {
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
                "score": 80,
                "factors": [
                    {"name": "weak credit", "points": 30, "detail": "credit band c"},
                    {
                        "name": "prior claims",
                        "points": 30,
                        "detail": "three in five years",
                    },
                    {"name": "aging asset", "points": 20, "detail": "22 year old home"},
                ],
                "rationale": "high risk home case with repeated claims",
            }
        ),
        "recommendation agent": json.dumps(
            {
                "decision": "deny",
                "reasons": ["score 80 is within the deny band"],
                "conditions": [],
            }
        ),
    },
    "CASE-003": {
        "intake agent": json.dumps(
            {
                "applicant_name": "Priya Natarajan",
                "date_of_birth": "1985-07-21",
                "product": "life",
                "coverage_amount": 500000,
                "stated_income": None,
                "missing_fields": [],
                "notes": ["stated income was blank"],
            }
        ),
        "enrichment agent": json.dumps(
            {
                "flags": ["no credit bureau record"],
                "notes": ["stated income not provided"],
            }
        ),
        "risk scoring agent": json.dumps(
            {
                "score": 45,
                "factors": [
                    {
                        "name": "no credit history",
                        "points": 25,
                        "detail": "bureau has no record",
                    },
                    {
                        "name": "unstated income",
                        "points": 20,
                        "detail": "applicant left income blank",
                    },
                ],
                "rationale": "unverifiable applicant, moderate risk",
            }
        ),
        "recommendation agent": json.dumps(
            {
                "decision": "refer",
                "reasons": ["no credit bureau record forces refer"],
                "conditions": ["verify identity and income manually"],
            }
        ),
    },
    "CASE-004": {
        "intake agent": json.dumps(
            {
                "applicant_name": "Elena Petrov",
                "date_of_birth": "1992-01-30",
                "product": "auto",
                "coverage_amount": 55000,
                "stated_income": 66000,
                "missing_fields": [],
                "notes": [],
            }
        ),
        "enrichment agent": json.dumps(
            {
                "flags": ["one prior claim", "vehicle is 6 years old"],
                "notes": ["credit band b"],
            }
        ),
        "risk scoring agent": json.dumps(
            {
                "score": 20,
                "factors": [
                    {"name": "minor prior claim", "points": 12, "detail": "windshield"},
                    {"name": "asset age", "points": 8, "detail": "6 year old vehicle"},
                ],
                "rationale": "acceptable auto risk",
            }
        ),
        "recommendation agent": json.dumps(
            {
                "decision": "approve",
                "reasons": ["score 20 is within the approve band"],
                "conditions": [],
            }
        ),
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixture underwriting cases through the pipeline."
    )
    parser.add_argument(
        "--case",
        default="all",
        help="CASE-XXX id or 'all' (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "scripted"],
        default="ollama",
        help="ollama runs live local inference, scripted is deterministic",
    )
    parser.add_argument(
        "--risk-failures",
        default="0",
        help="number of risk scoring calls to time out for the failure drill, "
        "or 'always' (default: 0, no injected failures)",
    )
    parser.add_argument(
        "--traces-dir",
        default=str(REPO_ROOT / "traces"),
        help="directory for trace JSON files, created if missing",
    )
    args = parser.parse_args(argv)

    try:
        risk_failures = parse_risk_failures(args.risk_failures)
    except ValueError:
        parser.error("--risk-failures must be an integer or 'always'")

    entries = load_applications(REPO_ROOT / "data" / "applications.json")
    selected = select_cases(entries, args.case)

    ollama: OllamaClient | None = None
    if args.backend == "ollama":
        ollama = OllamaClient()
        if not ollama.health():
            print(f"Ollama is not reachable at {ollama.host}.")
            print(
                "Start Ollama (ollama serve) and pull the model "
                "(ollama pull qwen3:8b), or pass --backend scripted "
                "for an offline run."
            )
            return 2

    traces_dir = Path(args.traces_dir)
    all_matched = True
    for entry in selected:
        raw = RawApplication(**entry["application"])
        base = (
            ollama if ollama is not None else ScriptedClient(SCRIPTED_RESPONSES[entry["case_id"]])
        )
        client = (
            TimeoutRiskClient(base, fail_first_n=risk_failures)
            if risk_failures
            else base
        )

        print()
        print(
            f"{entry['case_id']}: {raw.applicant_name} | "
            f"backend: {args.backend} | expected: {entry['expected_decision']}"
        )

        before = {p.name for p in traces_dir.glob("*.json")}
        record = run_case(raw, client, traces_dir=traces_dir)
        new_files = sorted({p.name for p in traces_dir.glob("*.json")} - before)
        if new_files:
            tracer = Tracer.load(traces_dir / new_files[-1])
            print(tracer.summary())

        final = record.recommendation.decision if record.recommendation else "none"
        matched = final == entry["expected_decision"]
        all_matched = all_matched and matched
        if record.status == "escalated":
            print(
                "NOTE: this case escalated (risk scoring kept failing), so code "
                "issued a referral without a model decision; against an approve "
                "expectation that MISMATCH is the designed outcome of the drill."
            )
        print(
            f"decision: {final} | expected: {entry['expected_decision']} | "
            f"{'MATCH' if matched else 'MISMATCH'}"
        )

    return 0 if all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
