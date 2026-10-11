"""Summarize traces/*.json into docs/cost_latency.md with measured numbers.

For each trace: token totals, agent-time latency (the sum of agent span
durations, excluding the pipeline parent and the pipeline_escalation span so
nothing is double counted), the pipeline wall time, and a hosted-equivalent
USD figure from underwriting.pricing. Actual marginal cost is $0.00 because
inference runs locally.
"""

import argparse
import json
from pathlib import Path

from underwriting.pricing import HOSTED_RATES_PER_MTOK, case_cost

REPO_ROOT = Path(__file__).resolve().parents[1]

# Spans that wrap other spans or code-only escalation bookkeeping, excluded
# from the per-case agent-time latency to avoid double counting.
NON_AGENT_SPANS = ("pipeline", "pipeline_escalation")

REALTIME_BUDGET_MS = 2000.0
PRICING_URL = "https://www.anthropic.com/pricing"
RATES_AS_OF = "2026-10"


def load_traces(traces_dir: Path) -> list[dict]:
    """Load every *.json trace file, sorted by file name."""
    records = []
    for path in sorted(Path(traces_dir).glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def trace_stats(data: dict) -> dict:
    """Measured numbers for one trace dict (as written by Tracer.dump)."""
    spans = data["spans"]
    agent_spans = [s for s in spans if s["agent"] not in NON_AGENT_SPANS]
    wall_ms = next(
        (
            float(s.get("duration_ms", 0.0))
            for s in spans
            if s["agent"] == "pipeline"
        ),
        0.0,
    )
    intake_span = next((s for s in spans if s["agent"] == "intake_agent"), None)
    applicant = ""
    if intake_span is not None and isinstance(intake_span.get("input"), dict):
        applicant = intake_span["input"].get("applicant_name", "")
    return {
        "case_id": data["case_id"],
        "applicant": applicant,
        "tokens_in": sum(int(s["usage"].get("input_tokens", 0)) for s in spans),
        "tokens_out": sum(int(s["usage"].get("output_tokens", 0)) for s in spans),
        "agent_ms": sum(float(s.get("duration_ms", 0.0)) for s in agent_spans),
        "wall_ms": wall_ms,
        "usd": case_cost(spans),
        "agent_durations": agent_spans,
    }


def slowest_agent(records: list[dict]) -> tuple[str, float, float]:
    """(agent, its total agent-span ms, total agent-span ms across all)."""
    totals: dict[str, float] = {}
    for data in records:
        for span in data["spans"]:
            if span["agent"] in NON_AGENT_SPANS:
                continue
            totals[span["agent"]] = totals.get(span["agent"], 0.0) + float(
                span.get("duration_ms", 0.0)
            )
    if not totals:
        return "", 0.0, 0.0
    agent = max(totals, key=lambda name: totals[name])
    return agent, totals[agent], sum(totals.values())


def build_report(records: list[dict]) -> str:
    """Render the loaded trace dicts into the markdown cost/latency report.

    One table row per trace plus an AVERAGE row, then the analysis prose:
    data provenance, batch and real-time verdicts against the quoting
    budget, the slowest agent, and the hosted-equivalent cost note.
    """
    stats = [trace_stats(data) for data in records]
    n = len(stats)
    avg = lambda key: sum(s[key] for s in stats) / n  # noqa: E731

    models = sorted(
        {
            span["usage"].get("model", "")
            for data in records
            for span in data["spans"]
            if span["usage"].get("model")
        }
    )
    if models == ["scripted"]:
        provenance = (
            "these numbers come from the scripted offline client (model label "
            "'scripted'), so latencies reflect in-process code, not a model"
        )
    else:
        provenance = (
            "these numbers come from live local "
            + ", ".join(models)
            + " inference through Ollama"
        )

    agent, agent_ms, total_agent_ms = slowest_agent(records)
    avg_agent_ms = avg("agent_ms")
    batch_hours_1k = avg_agent_ms * 1000 / 3_600_000.0
    if avg_agent_ms <= REALTIME_BUDGET_MS:
        realtime = (
            f"the measured average of {avg_agent_ms:.0f} ms per case is within "
            f"the 1-2 second ({REALTIME_BUDGET_MS:.0f} ms) real-time quoting "
            "budget, so the pipeline could also serve interactive quotes, not "
            "just batch"
        )
    else:
        realtime = (
            f"the measured average of {avg_agent_ms:.0f} ms per case exceeds "
            f"the 1-2 second ({REALTIME_BUDGET_MS:.0f} ms) real-time quoting "
            "budget, so quotes should be answered asynchronously while the "
            "pipeline keeps running in the background"
        )

    lines: list[str] = []
    lines.append("# Cost and latency: measured traces")
    lines.append("")
    lines.append(
        f"Source: {n} trace(s) from sequential four-call pipeline runs; "
        f"{provenance}."
    )
    lines.append(
        "Actual marginal cost per case: $0.00 (local inference). "
        "Hosted-equivalent USD applies HOSTED_RATES_PER_MTOK "
        f"(input ${HOSTED_RATES_PER_MTOK['input']:.2f}, output "
        f"${HOSTED_RATES_PER_MTOK['output']:.2f} per million tokens, "
        f"Sonnet-class hosted rates from {PRICING_URL}, as of {RATES_AS_OF})."
    )
    lines.append("")
    lines.append("| case_id | applicant | tokens in | tokens out | latency ms (agents) | wall ms (pipeline) | hosted-equivalent USD |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in stats:
        lines.append(
            f"| {s['case_id'][:12]} | {s['applicant']} | {s['tokens_in']} | "
            f"{s['tokens_out']} | {s['agent_ms']:.1f} | {s['wall_ms']:.1f} | "
            f"${s['usd']:.6f} |"
        )
    lines.append(
        f"| AVERAGE |  | {avg('tokens_in'):.0f} | {avg('tokens_out'):.0f} | "
        f"{avg_agent_ms:.1f} | {avg('wall_ms'):.1f} | ${avg('usd'):.6f} |"
    )
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append(
        f"- Batch: at a measured {avg_agent_ms:.0f} ms of agent time per case, "
        "a sequential 1,000 case overnight batch needs about "
        f"{batch_hours_1k:.1f} h of pure pipeline runtime, which fits an "
        "overnight window with room to spare."
    )
    lines.append(f"- Real-time verdict: {realtime}.")
    if agent:
        share = (agent_ms / total_agent_ms * 100.0) if total_agent_ms else 0.0
        lines.append(
            f"- Slowest agent: {agent} accounts for {agent_ms:.0f} ms of the "
            f"{total_agent_ms:.0f} ms total agent time ({share:.0f}%); that "
            "step is the first parallelization or optimization lever. (With "
            "real external data sources, enrichment usually dominates because "
            "it waits on bureau and registry calls; here every agent is a "
            "local model call, so the ranking comes purely from these traces.)"
        )
    lines.append(
        "- Cost: actual marginal cost is $0.00 per case because inference "
        "runs locally; the hosted-equivalent column shows what the same "
        "token volume would have cost at the cited hosted rates."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Load every trace and write the report; exit 1 when none are found."""
    parser = argparse.ArgumentParser(
        description="Summarize traces into docs/cost_latency.md."
    )
    parser.add_argument(
        "--traces-dir",
        default=str(REPO_ROOT / "traces"),
        help="directory containing trace JSON files",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "docs" / "cost_latency.md"),
        help="markdown report destination",
    )
    args = parser.parse_args(argv)

    traces_dir = Path(args.traces_dir)
    records = load_traces(traces_dir)
    if not records:
        print(
            f"No traces found in {traces_dir}: run scripts/run_cases.py first "
            "(it writes one trace JSON per case)."
        )
        return 1

    report = build_report(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out} from {len(records)} trace(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
