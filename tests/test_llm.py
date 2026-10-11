"""Tests for ScriptedClient routing and the parse_json_model repair contract."""

import pytest
from pydantic import BaseModel

from underwriting.llm import (
    AgentOutputError,
    LLMResponse,
    ScriptedClient,
    parse_json_model,
    strip_json_fences,
)


class _Box(BaseModel):
    """Minimal pydantic model used as the parse target."""

    value: int


def test_scripted_client_routes_by_marker_substring():
    """The first key found in the prompt wins and carries the fixed usage numbers."""
    client = ScriptedClient({"intake agent": "A", "risk scoring agent": "B"})
    out = client.complete("You are the risk scoring agent.", "score this case")
    assert out.text == "B"
    assert out.model == "scripted"
    assert out.input_tokens == 120
    assert out.output_tokens == 80


def test_scripted_client_first_match_wins():
    """With overlapping keys, dict order decides: the first match wins."""
    client = ScriptedClient({"agent": "first", "risk": "second"})
    out = client.complete("You are the risk scoring agent.", "score this case")
    assert out.text == "first"


def test_scripted_client_no_match_raises_key_error():
    """An unmatched prompt raises KeyError listing the available keys."""
    client = ScriptedClient({"intake agent": "A"})
    with pytest.raises(KeyError, match="no scripted response matches"):
        client.complete("You are the risk scoring agent.", "score this case")


def test_strip_json_fences_removes_fence_lines():
    """Leading json/plain fences and a trailing fence line are removed."""
    assert strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_json_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_json_fences('  {"a": 1}  ') == '{"a": 1}'


def test_parse_json_model_happy_path():
    """Valid JSON parses straight into the model class with no repair call."""
    client = ScriptedClient({"unused": "{}"})
    parsed = parse_json_model(client, "sys", "user", '{"value": 5}', _Box)
    assert parsed.value == 5


def test_parse_json_model_handles_fenced_json():
    """Fenced JSON parses after fence stripping, no repair call needed."""
    client = ScriptedClient({"unused": "{}"})
    text = '```json\n{"value": 7}\n```'
    parsed = parse_json_model(client, "sys", "user", text, _Box)
    assert parsed.value == 7


class _BadThenGood:
    """Fake client that returns a bad reply once, then the good reply."""

    def __init__(self, bad: str, good: str) -> None:
        """Queue the bad reply first, then the good one."""
        self.replies = [bad, good]
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        """Return replies in order, repeating the last one once exhausted."""
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model="fake")


def test_parse_json_model_repairs_once_when_first_reply_is_invalid():
    """An invalid first reply triggers exactly one repair call that succeeds."""
    client = _BadThenGood("not json at all", '{"value": 3}')
    first = client.complete("sys", "user")
    assert first.text == "not json at all"
    parsed = parse_json_model(client, "sys", "user", first.text, _Box)
    assert parsed.value == 3
    assert client.calls == 2


def test_parse_json_model_raises_after_failed_repair():
    """A second invalid reply raises AgentOutputError after the single repair attempt."""
    client = _BadThenGood("still bad", "also bad")
    first = client.complete("sys", "user")
    with pytest.raises(AgentOutputError, match="after repair"):
        parse_json_model(client, "sys", "user", first.text, _Box)
    assert client.calls == 2
