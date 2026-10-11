"""LLM clients and JSON repair parsing.

ScriptedClient is the deterministic offline client used by tests and demos.
OllamaClient talks to a local Ollama server over HTTP and maps its token
counts into LLMResponse. Both satisfy the same LLMClient protocol, so the
agents and the pipeline never know which one is behind them.
"""

import json
import os
import re
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError


class LLMResponse(BaseModel):
    """A model reply plus the token usage needed for the cost model."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class AgentOutputError(RuntimeError):
    """Raised when a model reply cannot be parsed into the expected schema."""


class LLMClient(Protocol):
    """Anything that can complete a prompt. Agents depend on this, not on a SDK."""

    def complete(self, system: str, user: str) -> LLMResponse:
        """Send one system+user prompt and return the model reply.

        Implementations return an LLMResponse carrying the reply text, token
        counts (0 when the backend does not report them), and the model name.
        They must be plain injectable objects so tests can substitute
        scripted or fault-injecting stand-ins. Implementations may raise on
        transport or routing failures; they never invent a reply.
        """
        ...


class ScriptedClient(LLMClient):
    """Deterministic offline LLMClient driven by marker keyed replies.

    Responses map a marker substring to a reply text. On complete(), the prompt
    is system + "\\n" + user and the FIRST key that appears in it wins, so
    system prompts that start with a marker like "You are the intake agent"
    route correctly with a key like "intake agent".
    """

    def __init__(
        self,
        responses: dict[str, str],
        input_tokens: int = 120,
        output_tokens: int = 80,
        model: str = "scripted",
    ) -> None:
        """Store the marker keyed replies and the fixed usage numbers per reply."""
        self.responses = dict(responses)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:
        """Return the first reply whose key appears in system + "\\n" + user."""
        prompt = system + "\n" + user
        for key, value in self.responses.items():
            if key in prompt:
                return LLMResponse(
                    text=value,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    model=self.model,
                )
        raise KeyError(
            f"no scripted response matches; available keys: {list(self.responses)}"
        )


class UsageRecordingClient(LLMClient):
    """Wraps any LLMClient and records every LLMResponse it returns.

    The pipeline uses one shared instance so each agent span can attach the
    usage of exactly the calls made during that span: drain() returns the
    responses recorded since the last drain and clears the buffer.
    """

    def __init__(self, inner: LLMClient) -> None:
        """Wrap inner; recording starts with an empty buffer."""
        self.inner = inner
        self.records: list[LLMResponse] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        """Delegate to inner, record the response, and return it unchanged."""
        response = self.inner.complete(system, user)
        self.records.append(response)
        return response

    def clear(self) -> None:
        """Drop all recorded responses."""
        self.records.clear()

    def drain(self) -> list[LLMResponse]:
        """Return the recorded responses and clear the buffer."""
        records = list(self.records)
        self.records.clear()
        return records


def strip_json_fences(text: str) -> str:
    """Remove a leading ```json or ``` fence line and a trailing ``` fence line."""
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


T = TypeVar("T", bound=BaseModel)


def parse_json_model(
    llm: LLMClient,
    system: str,
    user: str,
    text: str,
    model_cls: type[T],
) -> T:
    """Parse a model reply into model_cls, with exactly one repair attempt.

    On a JSON error or a schema validation error, the model is asked once to
    correct its reply given the validation error. A second failure raises
    AgentOutputError; no score or decision is ever invented here.
    """
    try:
        text2 = strip_json_fences(text)
        return model_cls.model_validate(json.loads(text2))
    except (json.JSONDecodeError, ValidationError) as exc:
        repair = llm.complete(
            system,
            user
            + "\n\nYour previous reply was not valid for the schema. The validation error was:\n"
            + str(exc)
            + "\nReturn only corrected JSON.",
        )
        try:
            repair_text = strip_json_fences(repair.text)
            return model_cls.model_validate(json.loads(repair_text))
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise AgentOutputError(
                f"invalid JSON from model after repair: {exc2}"
            ) from exc2


def complete_parsed(llm: LLMClient, system: str, user: str, model_cls: type[T]) -> T:
    """Complete a prompt and parse the reply into model_cls.

    One call to llm.complete() followed by parse_json_model(), so a reply
    that fails to parse gets exactly one repair attempt (see parse_json_model)
    and a still-invalid reply raises AgentOutputError; no value is ever
    invented here. Every agent uses this instead of hand-rolling the pair.
    """
    response = llm.complete(system, user)
    return parse_json_model(llm, system, user, response.text, model_cls)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Health checks must fail fast: this probes whether the server is up, not
# whether a cold model load has finished.
HEALTH_CHECK_TIMEOUT_SECONDS = 3.0


def strip_think(text: str) -> str:
    """Remove reasoning blocks from a model reply and trim whitespace.

    Handles any number of <think>...</think> blocks anywhere in the text
    plus a stray leading closing tag left by a truncated block.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    stripped = cleaned.lstrip()
    if stripped.startswith("</think>"):
        stripped = stripped[len("</think>"):]
    return stripped.strip()


def _default_host() -> str:
    """UNDERWRITING_OLLAMA_HOST, then OLLAMA_HOST, then the local default."""
    return (
        os.environ.get("UNDERWRITING_OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )


def _default_model() -> str:
    """UNDERWRITING_MODEL, then the qwen3:8b default."""
    return os.environ.get("UNDERWRITING_MODEL") or "qwen3:8b"


class OllamaClient(LLMClient):
    """LLMClient backed by a local Ollama server's /api/chat endpoint.

    The transport parameter exists for tests: pass an httpx.MockTransport to
    intercept HTTP calls; in normal use it stays None and httpx uses its
    default transport. Host and model default to the environment variables
    UNDERWRITING_OLLAMA_HOST (or OLLAMA_HOST) and UNDERWRITING_MODEL, falling
    back to http://127.0.0.1:11434 and qwen3:8b.
    """

    # The timeout=240.0 and keep_alive="30m" defaults are deliberate: a cold
    # start can spend minutes reloading the model into VRAM, and the long
    # keep-alive keeps the model resident across a whole run of cases.
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 240.0,
        temperature: float = 0.0,
        think: bool = False,
        keep_alive: str = "30m",
        transport: Any = None,
    ) -> None:
        """Store settings; a None host or model is resolved from the environment."""
        self.host = (host if host is not None else _default_host()).rstrip("/")
        self.model = model if model is not None else _default_model()
        self.timeout = timeout
        self.temperature = temperature
        self.think = think
        self.keep_alive = keep_alive
        self.transport = transport

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue an HTTP request against the configured host."""
        # httpx's top-level helpers do not take a transport, so an injected
        # transport (tests) goes through a short-lived httpx.Client instead.
        if self.transport is not None:
            with httpx.Client(transport=self.transport) as http:
                return http.request(method, self.host + path, **kwargs)
        return httpx.request(method, self.host + path, **kwargs)

    def complete(self, system: str, user: str) -> LLMResponse:
        """POST /api/chat and map the reply into an LLMResponse.

        Reasoning blocks are stripped from the content, ollama's
        prompt_eval_count/eval_count become the token counts (0 when
        missing), and the model name falls back to the configured model
        when the server omits it.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        response = self._request("POST", "/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = strip_think(message.get("content") or "")
        return LLMResponse(
            text=content,
            input_tokens=data.get("prompt_eval_count") or 0,
            output_tokens=data.get("eval_count") or 0,
            model=data.get("model") or self.model,
        )

    def health(self) -> bool:
        """True when the server answers GET /api/tags with a 200."""
        try:
            response = self._request(
                "GET", "/api/tags", timeout=HEALTH_CHECK_TIMEOUT_SECONDS
            )
            return response.status_code == 200
        except (httpx.HTTPError, OSError):
            # Transport-level failures only: a programming error must not
            # masquerade as "server unhealthy".
            return False


class TimeoutRiskClient(LLMClient):
    """Fault injector for failure drills: times out the risk scoring agent.

    Wraps any LLMClient. Calls whose prompt contains agent_marker raise
    TimeoutError for the first fail_first_n matching calls (use -1 for
    always); every other call, and later matching calls once the budget is
    spent, pass through to the inner client. The counter only advances on
    matching calls so fail_first_n means "first N risk scoring calls", not
    "first N calls of any agent". agent_marker defaults to the risk scoring
    marker the agents' system prompts actually start with; tests key on the
    same phrase.
    """

    def __init__(
        self,
        inner: LLMClient,
        fail_first_n: int,
        agent_marker: str = "risk scoring agent",
    ) -> None:
        """Wrap inner; fail_first_n bounds injected timeouts, -1 means always."""
        self.inner = inner
        self.fail_first_n = fail_first_n
        self.agent_marker = agent_marker
        self._calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        """Raise TimeoutError while the matching-call budget lasts, else pass through."""
        if self.agent_marker in (system + "\n" + user):
            if self.fail_first_n == -1 or self._calls < self.fail_first_n:
                self._calls += 1
                raise TimeoutError("simulated risk scoring timeout")
            self._calls += 1
        return self.inner.complete(system, user)
