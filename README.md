# insurance-orchestrator

A four-agent insurance underwriting pipeline lab (intake, enrichment, risk scoring, recommendation) that works on one shared case record. Runs are span traced, risk scoring retries transient failures before escalating to a code-issued referral, and inference runs locally through Ollama with a deterministic scripted fallback for offline work.

## Setup

Requirements: Python 3.12+ with [uv](https://docs.astral.sh/uv/), and Ollama with the model pulled (only for the ollama backend).

```bash
uv sync
ollama pull qwen3:8b
```

## Run

```bash
uv run pytest -q                                                # offline test suite: no network, no real LLM
uv run scripts/run_cases.py                                     # all four fixture cases on local Ollama, one trace JSON per run in traces/
uv run scripts/run_cases.py --risk-failures 2                   # failure drill: the risk agent times out twice, then the case recovers
uv run scripts/run_cases.py --risk-failures always              # failure drill: the risk agent always times out, the case escalates to a code-issued refer
uv run scripts/analyze_traces.py                                # summarize traces/ into docs/cost_latency.md (latency and hosted-equivalent cost)
uv run python scripts/build_submission.py                       # rebuild submission.pdf from docs, code, traces, and captured evidence
```

Handy flags for run_cases.py: `--case CASE-003` runs a single fixture, `--backend scripted` swaps the live model for the deterministic offline client (no Ollama needed, decisions always match expectations), and `--traces-dir` chooses where traces are written.

## Configuration

| Setting | Default | Override |
| --- | --- | --- |
| Ollama host | http://127.0.0.1:11434 | UNDERWRITING_OLLAMA_HOST or OLLAMA_HOST env var, or OllamaClient(host=...) |
| Model | qwen3:8b | UNDERWRITING_MODEL env var, or OllamaClient(model=...) |

Traces land in traces/ as one JSON file per case run; scripts/analyze_traces.py turns them into docs/cost_latency.md with per-case latency, token totals, and a hosted-equivalent USD estimate (the actual marginal cost of local inference is $0.00).
