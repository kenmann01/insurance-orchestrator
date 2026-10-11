"""Span tracing for the pipeline: one trace per case, one span per agent call.

A Tracer owns one trace. The caller enters `tracer.pipeline_span()` once and
opens agent spans inside it; agent spans reference the open pipeline span as
their parent, so a dumped trace shows the whole call tree for a case. Spans
are mutable: the caller sets input, output, usage, and status before exit.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from underwriting.llm import LLMResponse


class Span:
    """One timed operation in a trace, dict-friendly via to_dict()."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        case_id: str,
        agent: str,
        attempt: int,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.case_id = case_id
        self.agent = agent
        self.attempt = attempt
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.ended_at: str | None = None
        self.duration_ms: float = 0.0
        self.input: Any = None
        self.output: Any = None
        self.usage: dict = {"input_tokens": 0, "output_tokens": 0, "model": ""}
        self.status: str = "ok"
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "case_id": self.case_id,
            "agent": self.agent,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "usage": self.usage,
            "status": self.status,
            "error": self.error,
        }


def set_usage(span: Span, input_tokens: int, output_tokens: int, model: str) -> None:
    """Attach token totals and the model name to a span's usage dict."""
    span.usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }


def record_usage(span: Span, responses: list[LLMResponse]) -> None:
    """Sum usage over a list of LLMResponse objects onto a span.

    Tokens are summed across responses (a span may include several calls,
    for example the original call plus a repair call); the model name comes
    from the last response.
    """
    input_tokens = sum(r.input_tokens for r in responses)
    output_tokens = sum(r.output_tokens for r in responses)
    model = responses[-1].model if responses else ""
    set_usage(span, input_tokens, output_tokens, model)


class Tracer:
    """Collects spans for one case and can dump or summarize the trace."""

    def __init__(self, case_id: str) -> None:
        self.trace_id: str = uuid4().hex
        self.case_id = case_id
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.spans: list[Span] = []
        self._pipeline_span: Span | None = None

    @contextmanager
    def pipeline_span(self) -> Iterator[Span]:
        """Open the single parent span for the whole pipeline run."""
        span = Span(self.trace_id, uuid4().hex, None, self.case_id, "pipeline", 1)
        self.spans.append(span)
        self._pipeline_span = span
        try:
            with self._timed(span):
                yield span
        finally:
            self._pipeline_span = None

    @contextmanager
    def span(
        self, agent: str, *, input: Any = None, attempt: int = 1
    ) -> Iterator[Span]:
        """Open an agent span parented on the currently open pipeline span."""
        parent = self._pipeline_span
        span = Span(
            self.trace_id,
            uuid4().hex,
            parent.span_id if parent is not None else None,
            self.case_id,
            agent,
            attempt,
        )
        if input is not None:
            span.input = input
        self.spans.append(span)
        with self._timed(span):
            yield span

    @contextmanager
    def _timed(self, span: Span) -> Iterator[Span]:
        start = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            span.ended_at = datetime.now(timezone.utc).isoformat()
            span.duration_ms = round((time.perf_counter() - start) * 1000.0, 1)
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        span.ended_at = datetime.now(timezone.utc).isoformat()
        span.duration_ms = round((time.perf_counter() - start) * 1000.0, 1)

    def dump(self, path: str | Path) -> Path:
        """Write the trace as indented JSON and return the path written."""
        target = Path(path)
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "created_at": self.created_at,
            "spans": [span.to_dict() for span in self.spans],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def summary(self) -> str:
        """Fixed-width ASCII table: one row per span plus a TOTAL row."""
        header = (
            f"{'AGENT':<24} {'ATT':>4} {'STATUS':<10} {'MS':>9} "
            f"{'TOKENS IN':>10} {'TOKENS OUT':>11}"
        )
        rule = "=" * len(header)
        thin = "-" * len(header)
        lines = [rule, header, thin]
        total_in = 0
        total_out = 0
        total_ms = 0.0
        for span in self.spans:
            tokens_in = int(span.usage.get("input_tokens", 0))
            tokens_out = int(span.usage.get("output_tokens", 0))
            total_in += tokens_in
            total_out += tokens_out
            total_ms += span.duration_ms
            lines.append(
                f"{span.agent:<24} {span.attempt:>4} {span.status:<10} "
                f"{span.duration_ms:>9.1f} {tokens_in:>10} {tokens_out:>11}"
            )
        lines.append(thin)
        lines.append(
            f"{'TOTAL':<24} {'':>4} {'':<10} {round(total_ms, 1):>9.1f} "
            f"{total_in:>10} {total_out:>11}"
        )
        lines.append(rule)
        return "\n".join(lines)
