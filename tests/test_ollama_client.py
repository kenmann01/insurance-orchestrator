"""Tests for the Ollama HTTP client using httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from underwriting.llm import OllamaClient, _default_host, _default_model, strip_think


def chat_handler(payload_json: dict):
    """Build a MockTransport handler echoing a canned /api/chat payload."""
    def handler(request: httpx.Request) -> httpx.Response:
        """Assert the chat path and return the canned payload plus the sent request."""
        assert request.url.path == "/api/chat"
        sent = json.loads(request.content)
        return httpx.Response(200, json={**payload_json, "request": sent})

    return handler


def test_strip_think_removes_blocks_and_stray_closing_tag():
    """Think blocks, a stray closing tag, and surrounding whitespace are all removed."""
    text = "<think>reasoning here</think>\n{\"a\": 1}"
    assert strip_think(text) == '{"a": 1}'
    text2 = "<think>one</think>middle<think>two</think>end"
    assert strip_think(text2) == "middleend"
    text3 = "</think>{\"b\": 2}"
    assert strip_think(text3) == '{"b": 2}'
    text4 = "  plain reply  "
    assert strip_think(text4) == "plain reply"


def test_complete_strips_think_and_maps_token_counts():
    """A think-wrapped reply is stripped and ollama's token counts map into LLMResponse."""
    canned = {
        "model": "qwen3:8b",
        "message": {
            "content": "<think>let me think it through</think>\n{\"score\": 10}"
        },
        "prompt_eval_count": 25,
        "eval_count": 41,
    }
    client = OllamaClient(transport=httpx.MockTransport(chat_handler(canned)))
    response = client.complete("You are the intake agent", "Application JSON: {}")
    assert response.text == '{"score": 10}'
    assert response.input_tokens == 25
    assert response.output_tokens == 41
    assert response.model == "qwen3:8b"


def test_complete_defaults_missing_counts_and_model(monkeypatch):
    """Missing token counts become 0 and a missing model falls back to the configured one."""
    # Pin the default model fallback: on a dev box UNDERWRITING_MODEL may be
    # set, and this test asserts the qwen3:8b fallback specifically.
    monkeypatch.delenv("UNDERWRITING_MODEL", raising=False)
    monkeypatch.delenv("UNDERWRITING_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    canned = {"message": {"content": "ok"}}
    client = OllamaClient(
        host="http://127.0.0.1:11434/",
        transport=httpx.MockTransport(chat_handler(canned)),
    )
    response = client.complete("system", "user")
    assert response.text == "ok"
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.model == "qwen3:8b"  # falls back to the configured model


def test_complete_payload_shape():
    """The /api/chat payload carries stream, think, keep_alive, options, and messages."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the request JSON and answer with a minimal canned reply."""
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"model": "qwen3:8b", "message": {"content": "x"}},
        )

    client = OllamaClient(
        model="qwen3:8b",
        temperature=0.2,
        think=False,
        keep_alive="45m",
        transport=httpx.MockTransport(handler),
    )
    client.complete("sys prompt", "usr prompt")
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["keep_alive"] == "45m"
    assert captured["options"]["temperature"] == 0.2
    assert captured["model"] == "qwen3:8b"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "sys prompt"
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "usr prompt"


def test_health_true_on_200_and_false_on_error():
    """health() is True on a 200 from /api/tags and False on a connection error."""
    def ok_handler(request: httpx.Request) -> httpx.Response:
        """Answer GET /api/tags with 200 and an empty model list."""
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    good = OllamaClient(transport=httpx.MockTransport(ok_handler))
    assert good.health() is True

    def dead_handler(request: httpx.Request) -> httpx.Response:
        """Simulate a dead server by raising a connect error."""
        raise httpx.ConnectError("connection refused")

    dead = OllamaClient(transport=httpx.MockTransport(dead_handler))
    assert dead.health() is False


def test_default_host_and_model_fallback_order(monkeypatch):
    """UNDERWRITING_OLLAMA_HOST beats OLLAMA_HOST beats the default; UNDERWRITING_MODEL beats qwen3:8b."""
    # Implemented order: UNDERWRITING_OLLAMA_HOST beats OLLAMA_HOST beats the
    # built-in default; UNDERWRITING_MODEL beats the built-in model.
    monkeypatch.delenv("UNDERWRITING_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("UNDERWRITING_MODEL", raising=False)
    assert _default_host() == "http://127.0.0.1:11434"
    assert _default_model() == "qwen3:8b"

    monkeypatch.setenv("OLLAMA_HOST", "http://fallback:11434")
    assert _default_host() == "http://fallback:11434"
    monkeypatch.setenv("UNDERWRITING_OLLAMA_HOST", "http://primary:11434")
    assert _default_host() == "http://primary:11434"
    monkeypatch.setenv("UNDERWRITING_MODEL", "llama3:8b")
    assert _default_model() == "llama3:8b"

    # The constructor uses the same resolution when host/model are omitted.
    client = OllamaClient()
    assert client.host == "http://primary:11434"
    assert client.model == "llama3:8b"

    monkeypatch.delenv("UNDERWRITING_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("UNDERWRITING_MODEL", raising=False)
    client = OllamaClient()
    assert client.host == "http://fallback:11434"
    assert client.model == "qwen3:8b"


def test_http_error_raises():
    """A 500 reply raises httpx.HTTPStatusError instead of being swallowed."""
    def fail_handler(request: httpx.Request) -> httpx.Response:
        """Return a 500 response with an error body."""
        return httpx.Response(500, json={"error": "boom"})

    client = OllamaClient(transport=httpx.MockTransport(fail_handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("system", "user")


def test_complete_without_injected_transport_uses_plain_request(monkeypatch):
    """Without a transport the client issues the request via httpx.request directly."""
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            """No-op success."""

        def json(self):
            """Minimal /api/chat body with token counts."""
            return {
                "model": "qwen3:8b",
                "message": {"content": "plain path reply"},
                "prompt_eval_count": 7,
                "eval_count": 3,
            }

    def fake_request(method, url, **kwargs):
        """Record the call and answer without touching the network."""
        seen["method"] = method
        seen["url"] = url
        seen["payload"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setattr(httpx, "request", fake_request)
    client = OllamaClient()
    response = client.complete("sys prompt", "usr prompt")
    assert response.text == "plain path reply"
    assert (response.input_tokens, response.output_tokens) == (7, 3)
    assert seen["method"] == "POST"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["payload"]["messages"][1]["content"] == "usr prompt"
