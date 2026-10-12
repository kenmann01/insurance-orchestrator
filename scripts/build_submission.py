"""Build the submission PDF for the Week 8 lab.

Assembles the 8 deliverable sections into one print-styled HTML file at the
repo root, then prints it to submission.pdf with headless Edge. Missing
evidence files render as amber PENDING boxes instead of failing the build;
docs/ and src/ inputs are mandatory and fail loudly when absent.

    uv run python scripts/build_submission.py
"""

import ast
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEYWORDS = {
    "def", "class", "return", "if", "elif", "else", "for", "while", "in", "not",
    "and", "or", "import", "from", "as", "with", "async", "await", "try", "except",
    "raise", "None", "True", "False", "self", "yield", "lambda", "pass", "break",
    "continue", "is", "assert", "finally", "del",
}

AGENTS = (
    "intake_agent",
    "enrichment_agent",
    "risk_scoring_agent",
    "recommendation_agent",
)


def read(rel: str) -> str:
    """Read a repo-relative mandatory file and drop trailing whitespace.

    A missing docs/ or src/ input raises FileNotFoundError here on purpose:
    the build fails loudly instead of shipping a page with a hole in it.
    """
    return (ROOT / rel).read_text(encoding="utf-8").rstrip()


def highlight_python(code: str) -> str:
    """Wrap Python keywords, strings, and comments in highlight spans."""
    out = []
    for line in code.splitlines():
        escaped = html.escape(line, quote=False)
        parts = re.split(r"(#.*$)", escaped, maxsplit=1)
        tokens = []
        for i, part in enumerate(parts):
            if i == 1:
                tokens.append(f'<span class="c">{part}</span>')
                continue
            part = re.sub(r'(".*?"|\'.*?\')', r'<span class="s">\1</span>', part)
            for kw in KEYWORDS:
                part = re.sub(rf"\b({kw})\b(?![^<]*</span>)", r'<span class="k">\1</span>', part)
            tokens.append(part)
        out.append("".join(tokens))
    return "\n".join(out)


def code_block(source: str) -> str:
    """Render Python source as a highlighted code block."""
    return f'<pre class="code">{highlight_python(source)}</pre>'


def term_block(source: str) -> str:
    """Render captured terminal output."""
    return f'<pre class="term">{html.escape(source)}</pre>'


def doc_block(source: str) -> str:
    """Render a Markdown document as preformatted text."""
    return f'<pre class="doc">{html.escape(source)}</pre>'


def file_label(rel: str) -> str:
    """Render the repo-relative path shown above a block."""
    return f'<div class="file">{html.escape(rel)}</div>'


def code_item(rel: str) -> str:
    """Load one Python file and render it as a labeled code block."""
    return file_label(rel) + code_block(read(rel))


def doc_item(rel: str) -> str:
    """Load one Markdown file and render it as a labeled document block."""
    return file_label(rel) + doc_block(read(rel))


def pending_box(rel: str, command: str) -> str:
    """Render the amber PENDING box for one missing evidence file."""
    return (
        f'<div class="pending"><b>PENDING:</b> {html.escape(rel)} is missing from '
        "this checkout, so the exhibit cannot be shown. Produce it, then rebuild "
        f'this PDF, with: <code>{html.escape(command)}</code></div>'
    )


def try_term(rel: str, command: str) -> str:
    """Render captured terminal output, or a PENDING box naming the producer."""
    if (ROOT / rel).exists():
        return file_label(rel) + term_block(read(rel))
    return pending_box(rel, command)


def class_excerpt(rel: str, class_name: str) -> str:
    """Slice exactly one top-level class out of a source file.

    The excerpt starts at the class line and ends at the next top-level
    class or def, or at end of file, so only that class's source lines are
    kept.
    """
    lines = read(rel).splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"class {class_name}")),
        None,
    )
    if start is None:
        raise ValueError(f"class {class_name} not found in {rel}")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith(("class ", "def ")):
            end = i
            break
    return "\n".join(lines[start:end]).rstrip()


def retry_excerpt(rel: str, start_marker: str) -> str:
    """Slice a source file from start_marker to end of file, syntax checked.

    The retry and escalation handling lives after call_with_retry, so the
    cut to end of file covers the whole retry-then-escalate story. ast.parse
    guards the cut; a slice that does not parse falls back to the whole file.
    """
    source = read(rel)
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if start_marker in line), None)
    if start is None:
        return source
    excerpt = "\n".join(lines[start:]).rstrip()
    try:
        ast.parse(excerpt)
    except SyntaxError:
        return source
    return excerpt


def trace_summaries() -> list[dict]:
    """Parse every trace JSON and compute one summary row per trace."""
    rows = []
    for path in sorted((ROOT / "traces").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        spans = data["spans"]

        def span_output(agent: str) -> dict:
            for s in spans:
                if s["agent"] == agent:
                    return s.get("output") or {}
            return {}

        durations = {agent: 0.0 for agent in AGENTS}
        tokens_in = tokens_out = 0
        statuses: set[str] = set()
        for s in spans:
            if s["agent"] in durations:
                durations[s["agent"]] += float(s.get("duration_ms") or 0.0)
            usage = s.get("usage") or {}
            tokens_in += int(usage.get("input_tokens") or 0)
            tokens_out += int(usage.get("output_tokens") or 0)
            statuses.add(s.get("status") or "unknown")

        recommendation = span_output("recommendation_agent")
        risk = span_output("risk_scoring_agent")
        rows.append(
            {
                "file": data.get("trace_id", path.stem)[:8],
                "applicant": span_output("intake_agent").get("applicant_name") or "unknown",
                "decision": recommendation.get("decision") or "none (escalated)",
                "score": risk.get("score") if risk.get("score") is not None else "n/a",
                "band": span_output("enrichment_agent").get("credit_band") or "n/a",
                "durations": durations,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "statuses": "/".join(sorted(statuses)),
                "path": path.relative_to(ROOT).as_posix(),
                "json": data,
            }
        )
    return rows


def trace_table(rows: list[dict]) -> str:
    """Render the per-trace summary table computed from the JSON files."""
    header = (
        "<tr><th>Trace</th><th>Applicant</th><th>Decision</th><th>Score</th>"
        "<th>Band</th><th>Intake ms</th><th>Enrich ms</th><th>Risk ms</th>"
        "<th>Rec ms</th><th>Tokens in</th><th>Tokens out</th><th>Span statuses</th></tr>"
    )
    body = []
    for r in rows:
        cells = [
            r["file"],
            r["applicant"],
            r["decision"],
            str(r["score"]),
            str(r["band"]),
            f"{r['durations']['intake_agent']:,.0f}",
            f"{r['durations']['enrichment_agent']:,.0f}",
            f"{r['durations']['risk_scoring_agent']:,.0f}",
            f"{r['durations']['recommendation_agent']:,.0f}",
            f"{r['tokens_in']:,}",
            f"{r['tokens_out']:,}",
            r["statuses"],
        ]
        body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    return f'<table class="tbl">{header}{"".join(body)}</table>'


def section(num: int, title: str, body: str) -> str:
    """Wrap one numbered deliverable section, starting on a fresh page."""
    return f"<section><h2>{num}. {html.escape(title)}</h2>{body}</section>"


TOC: list[tuple[str, str]] = [
    ("1. State and memory design decision", "docs/design.md"),
    (
        "2. Standalone agent unit tests and passing output",
        "tests/test_intake.py, test_enrichment.py, test_risk.py, "
        "test_recommendation.py, evidence/pytest.txt",
    ),
    (
        "3. Full pipeline code and the state mechanism",
        "src/underwriting/pipeline.py, src/underwriting/models.py",
    ),
    (
        "4. Tracing implementation",
        "src/underwriting/tracing.py plus the UsageRecordingClient excerpt "
        "from src/underwriting/llm.py",
    ),
    ("5. Full traces for the mock applications", "traces/*.json, four captured Ollama runs"),
    ("6. Cost and latency analysis", "docs/cost_latency.md"),
    (
        "7. Simulated failure: retry then escalate",
        "TimeoutRiskClient excerpt, the retry and escalation slice of pipeline.py, "
        "tests/test_failure.py, evidence/failure_recovery.txt, evidence/failure_escalation.txt",
    ),
    (
        "8. Passing pipeline run (CI)",
        "evidence/ci-run.txt plus evidence/pytest.txt and the GitHub Actions page",
    ),
]


def build_cover() -> str:
    """Render the title page with its own A4 height and a mapped table of contents."""
    toc_rows = "".join(
        f"<tr><td>{html.escape(num_title)}</td><td>{html.escape(maps_to)}</td></tr>"
        for num_title, maps_to in TOC
    )
    return (
        '<div class="cover">'
        "<h1>Multi-Agent Underwriting Pipeline</h1>"
        '<div class="sub">Week 8 Lab Submission</div>'
        '<div class="line">Four agents (intake, enrichment, risk scoring, recommendation) '
        "on a shared case record, with tracing, retry-then-escalate failure handling, "
        "and CI</div>"
        '<div class="line repo">github.com/kenmann01/insurance-orchestrator</div>'
        '<div class="toc-title">Contents</div>'
        f"<table>{toc_rows}</table>"
        "</div>"
    )


CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #111; margin: 0; }
.cover { height: 252mm; display: flex; flex-direction: column; justify-content: center; }
.cover h1 { font-size: 28pt; margin: 0 0 8px; }
.cover .sub { font-size: 13pt; color: #555; margin-bottom: 26px; }
.cover .line { font-size: 11.5pt; color: #333; margin: 0 0 10px; max-width: 168mm; }
.cover .repo { font-family: Consolas, monospace; color: #1f6feb; margin-bottom: 40px; }
.cover .toc-title { font-size: 13pt; font-weight: 600; color: #1f6feb; margin: 0 0 6px; }
.cover table { border-collapse: collapse; font-size: 10.5pt; max-width: 175mm; }
.cover td { padding: 4px 14px 4px 0; vertical-align: top; }
.cover td:first-child { color: #111; white-space: nowrap; }
.cover td:last-child { color: #555; }
h2 { font-size: 15.5pt; border-bottom: 2.5px solid #1f6feb; padding-bottom: 5px; margin: 0 0 14px; }
section { page-break-before: always; }
pre { margin: 8px 0 14px; padding: 10px 12px; border-radius: 6px; font-size: 8.1pt; line-height: 1.38; }
pre.term { background: #0d1117; color: #e6edf3; white-space: pre-wrap; word-break: break-word; font-family: Consolas, 'Courier New', monospace; }
pre.code { background: #f6f8fa; border: 1px solid #d0d7de; white-space: pre-wrap; word-break: break-word; font-family: Consolas, 'Courier New', monospace; }
pre.doc { background: #fffdf5; border: 1px solid #e3dcc3; white-space: pre-wrap; word-break: break-word; font-family: Consolas, 'Courier New', monospace; font-size: 8.4pt; }
pre.code .k { color: #cf222e; font-weight: 600; }
pre.code .s { color: #0a3069; }
pre.code .c { color: #6e7781; font-style: italic; }
.file { font-family: Consolas, monospace; font-size: 9pt; color: #1f6feb; margin: 12px 0 0; page-break-after: avoid; break-after: avoid; }
.note { font-size: 10pt; color: #444; margin: 8px 0 2px; }
.pending { background: #fff8e1; border: 1.5px solid #e0a800; color: #6b4d00; padding: 10px 12px; border-radius: 6px; font-size: 9.5pt; margin: 10px 0 14px; page-break-inside: avoid; }
.pending code { font-family: Consolas, monospace; background: #ffefb0; padding: 1px 5px; border-radius: 3px; }
table.tbl { border-collapse: collapse; font-size: 8pt; margin: 10px 0 16px; width: 100%; }
.tbl th, .tbl td { border: 1px solid #d0d7de; padding: 3px 6px; text-align: left; }
.tbl th { background: #f6f8fa; }
.url { font-family: Consolas, monospace; font-size: 9.5pt; margin: 6px 0 12px; }
.url a { color: #1f6feb; }
"""


def build_html() -> str:
    """Assemble the eight deliverable sections into one HTML document."""
    p = [
        "<html><head><meta charset='utf-8'>"
        "<title>Multi-Agent Underwriting Pipeline, Week 8 Lab Submission</title>"
        f"<style>{CSS}</style></head><body>",
        build_cover(),
    ]

    # 1 state and memory design decision
    p.append(section(1, "State and memory design decision", doc_item("docs/design.md")))

    # 2 agent unit tests and passing output
    body2 = (
        '<div class="note">Each agent carries its own unit test file against the '
        "scripted offline client, so no test needs a live model.</div>"
        + code_item("tests/test_intake.py")
        + code_item("tests/test_enrichment.py")
        + code_item("tests/test_risk.py")
        + code_item("tests/test_recommendation.py")
        + try_term("evidence/pytest.txt", "uv run pytest -q")
    )
    p.append(section(2, "Standalone agent unit tests and passing output", body2))

    # 3 full pipeline code and the state mechanism
    body3 = (
        '<div class="note">The state mechanism is one shared CaseRecord: '
        "models.py defines the record and the per-agent payloads, and pipeline.py "
        "walks intake, enrichment, risk scoring, recommendation, mutating the "
        "same record in sequence inside a traced pipeline span.</div>"
        + code_item("src/underwriting/pipeline.py")
        + code_item("src/underwriting/models.py")
    )
    p.append(section(3, "Full pipeline code and the state mechanism", body3))

    # 4 tracing implementation
    body4 = (
        '<div class="note">Every agent call runs inside a Tracer span; the shared '
        "UsageRecordingClient sits between the pipeline and the backend so each "
        "span attaches the token usage of exactly its own calls.</div>"
        + code_item("src/underwriting/tracing.py")
        + file_label("src/underwriting/llm.py, UsageRecordingClient excerpt")
        + code_block(class_excerpt("src/underwriting/llm.py", "UsageRecordingClient"))
    )
    p.append(section(4, "Tracing implementation", body4))

    # 5 full traces for the mock applications
    rows = trace_summaries()
    body5 = (
        '<div class="note">Summary computed by this builder directly from the four '
        "traces/*.json files (plain json parsing, no hand transcription). Full "
        "pretty JSON per trace follows below.</div>"
        + trace_table(rows)
    )
    for r in rows:
        body5 += (
            f'<div class="file">{html.escape(r["path"])}: '
            f'{html.escape(r["applicant"])} ({html.escape(r["decision"])})</div>'
            + f'<pre class="code">{html.escape(json.dumps(r["json"], indent=2))}</pre>'
        )
    p.append(section(5, "Full traces for the mock applications", body5))

    # 6 cost and latency analysis
    p.append(section(6, "Cost and latency analysis", doc_item("docs/cost_latency.md")))

    # 7 simulated failure: retry then escalate
    body7 = (
        '<div class="note">The fault injector wraps any LLMClient and raises '
        "TimeoutError on the risk scoring calls while its failure budget lasts.</div>"
        + file_label("src/underwriting/llm.py, TimeoutRiskClient excerpt")
        + code_block(class_excerpt("src/underwriting/llm.py", "TimeoutRiskClient"))
        + '<div class="note">The retry loop and the escalation: call_with_retry owns '
        "attempt numbering and backoff, and when the budget is spent the pipeline "
        "refers the case with a fixed reason instead of letting an agent invent "
        "a decision.</div>"
        + file_label("src/underwriting/pipeline.py, retry and escalation excerpt "
                     "(from def call_with_retry to end of file)")
        + code_block(retry_excerpt("src/underwriting/pipeline.py", "def call_with_retry"))
        + '<div class="note">The failure behavior is pinned by unit tests with a '
        "short backoff schedule and a fake sleep.</div>"
        + code_item("tests/test_failure.py")
        + '<div class="note">Two captured drills against local Ollama: recovery '
        "(two injected timeouts, then the case completes) and escalation (the "
        "risk agent always times out, so code issues the referral).</div>"
        + try_term(
            "evidence/failure_recovery.txt",
            "uv run python scripts/run_cases.py --backend ollama --case CASE-004 --risk-failures 2",
        )
        + try_term(
            "evidence/failure_escalation.txt",
            "uv run python scripts/run_cases.py --backend ollama --case CASE-004 --risk-failures always",
        )
    )
    p.append(section(7, "Simulated failure: retry then escalate", body7))

    # 8 passing pipeline run (CI)
    body8 = (
        '<div class="note">The GitHub Actions workflow runs the same offline suite '
        "on every push; the latest run is green. The 61-test local capture of the "
        "same suite is evidence/pytest.txt in section 2, and coverage is enforced "
        "at a 97 percent floor in pyproject.</div>"
        + try_term("evidence/coverage.txt", "uv run pytest -q --cov --cov-report=term-missing")
        + try_term("evidence/ci-run.txt", "gh run list --repo kenmann01/insurance-orchestrator --limit 1")
        + '<div class="url"><a href="https://github.com/kenmann01/insurance-orchestrator/actions">'
        "https://github.com/kenmann01/insurance-orchestrator/actions</a></div>"
    )
    p.append(section(8, "Passing pipeline run (CI)", body8))

    p.append("</body></html>")
    return "".join(p)


def main() -> None:
    """Write submission.html and print it to submission.pdf with headless Edge."""
    html_path = ROOT / "submission.html"
    html_path.write_text(build_html(), encoding="utf-8")

    pdf_path = ROOT / "submission.pdf"
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    subprocess.run(
        [
            edge,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    size_bytes = pdf_path.stat().st_size
    print(f"wrote {html_path.name} and {pdf_path.name} ({size_bytes / 1024:.0f} KB, "
          "A4 pages; size is the health hint, page count not parsed)")
    if size_bytes < 20 * 1024:
        print(f"ERROR: submission.pdf is only {size_bytes / 1024:.1f} KB; "
              "Edge likely failed to print the document.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
