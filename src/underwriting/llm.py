"""LLM client protocol, a scripted offline client, and JSON repair parsing.

Nothing in this module touches the network: tests and demos use ScriptedClient,
and later commits can add a real client (for example an Ollama HTTP client)
that satisfies the same LLMClient protocol.
"""

import json
from typing import Protocol, TypeVar

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

    def complete(self, system: str, user: str) -> LLMResponse: ...


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
        self.responses = dict(responses)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:
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
        self.inner = inner
        self.records: list[LLMResponse] = []

    def complete(self, system: str, user: str) -> LLMResponse:
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
