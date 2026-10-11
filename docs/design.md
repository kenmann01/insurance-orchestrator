# Design: orchestration, state, and memory

This document records the structural decisions for the four agent pipeline before any
agent code is written: why it is a fixed workflow, why state is a shared case record
rather than message passing, what memory the system keeps, who may read and write which
section, how runs are traced, how failures are handled, and how cost is modeled.

## Architecture: a fixed workflow (prompt chaining), not an open agent loop

The pipeline is a prompt chaining workflow in the sense of Anthropic's engineering guide
"Building Effective Agents", which is the reference for choosing a workflow over a
dynamic agent. The steps are known in advance and always run in the same order: intake,
enrichment, risk scoring, recommendation. The guide's core advice is to find the
simplest pattern that works and to reach for autonomous, loop driven agents only when
the path through the task cannot be planned ahead of time. An underwriting case has
exactly one path, so orchestration here is four function calls in a fixed sequence plus
a small amount of glue for tracing and failure handling.

We explicitly reject agent frameworks such as LangGraph and CrewAI. Coordination in this
lab is four function calls with typed arguments; a framework would hide the handoff
between steps (which section is read, which field is written, when a step failed) behind
a graph or crew abstraction while adding nothing the lab needs. Keeping the handoff in
plain code also means a reviewer can see the whole control flow on one screen.

## State: a shared case record, not pure message passing

All four agents operate on one shared `CaseRecord` instead of passing messages. The
justification:

- A case file is the real underwriting object. Underwriters work from a single file
  that accumulates sections, and the lab mirrors that domain object directly.
- The recommendation genuinely needs intake, enrichment, and risk together. With pure
  message passing, each step would forward everything it received, and by the last step
  the payload would have grown into one deeply nested blob anyway: a case record with a
  worse name.
- A failed step leaves the earlier sections intact in the record, so a human reviewer
  can open the record or the trace and still see intake and enrichment even when risk
  scoring died.

The risk of a shared structure is hidden coupling, so the discipline is explicit: each
agent may read the sections it needs and may write ONLY its own field. Concretely, the
agents are pure functions that receive what they need and return a result; the pipeline
orchestrator performs every write, assigning each agent's result to its own field
immediately after the call returns: `case.intake` comes only from intake_agent's
output, `case.enrichment` only from enrichment_agent's, and the same rule holds for
`case.risk` and `case.recommendation`. The orchestrator also touches `case.status`
and, on escalation, forces the recommendation in the way described under the failure
policy.

## Memory scope: per case only

Memory is scoped to a single case. There is no cross-case store, no vector database, and
no learning from previous cases. The record lives in memory during a run and is written
to `traces/<trace_id>.json` so a run can be replayed from disk. Replaying a trace is the
only way a past run influences later inspection, and it is a manual, local operation.

## Read/write table

| Agent | Reads | Writes (its own field only) |
| --- | --- | --- |
| intake_agent | raw application | `case.intake`: normalized fields, missing-field list, notes. Never invents values for blank fields. |
| enrichment_agent | `case.intake` plus a local bureau lookup | `case.enrichment`: credit band, prior-claim count, asset age, flags, notes. Bureau facts are set by code, never by the model. |
| risk_scoring_agent | `case.intake`, `case.enrichment` | `case.risk`: integer 0-100 score, factor list, rationale. |
| recommendation_agent | `case.intake`, `case.enrichment`, `case.risk` | `case.recommendation`: approve/deny/refer, reasons, conditions. When risk was escalated the pipeline forces refer without invoking this agent. |

## Tracing plan

There is one span per agent call plus a parent pipeline span. Span fields:

- `trace_id`, `span_id`, `parent_span_id`
- `case_id`
- `agent`
- `attempt` (1 for the first try, incremented on retries)
- `started_at`, `ended_at`: ISO-8601 UTC timestamps
- `duration_ms`: measured with `perf_counter` (monotonic, so it is safe for durations)
- `input`, `output`: the prompt parts and the parsed result (output stays null on failure; the failure text lives in `error`)
- `usage`: `input_tokens`, `output_tokens`, `model`
- `status`: one of `ok`, `error`, `retry`, `escalated`
- `error`: the error message when status is not `ok`, else null

## Failure policy: retry, then escalate

The risk scoring agent is retried up to 3 attempts (1 initial + 2 retries) with short
backoff between attempts. Each failed attempt is recorded as a span with status
`retry`, so the trace shows exactly what the model produced before it failed.

On exhaustion the case is escalated rather than faked:

- `case.status` becomes `escalated`
- `case.risk` stays `None`: no score is ever fabricated
- the recommendation is forced to `refer` with an explanatory reason, without invoking
  the recommendation agent, because issuing approve or deny without a score would be
  invention rather than inference

## Cost model

Token usage is captured from real local inference (Ollama, qwen3:8b). The actual
marginal cost of a run is $0.00 because the model runs locally. For comparison, a
hosted-equivalent cost is computed from the measured token counts at the provider's
published API rates, and the rate table used is cited next to the number. This keeps
two questions separate: what did the run spend (nothing) and what would the same token
volume cost in production (a real number a reviewer can verify).
