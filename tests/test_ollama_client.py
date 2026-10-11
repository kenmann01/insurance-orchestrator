"""Tests for the Ollama HTTP client using httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from underwriting.llm import OllamaClient, strip_think


def chat_handler(payload_json: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        sent = json.loads(request.content)
        return httpx.Response(200, json={**payload_json, "request": sent})

    return handler


def test_strip_think_removes_blocks_and_stray_closing_tag():
    text = "<think>reasoning here</think>\n{\"a\": 1}"
    assert strip_think(text) == '{"a": 1}'
    text2 = "<think>one</think>middle<think>two</think>end"
    assert strip_think(text2) == "middleend"
    text3 = "</think>{\"b\": 2}"
    assert strip_think(text3) == '{"b": 2}'
    text4 = "  plain reply  "
    assert strip_think(text4) == "plain reply"


def test_complete_strips_think_and_maps_token_counts():
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


def test_complete_defaults_missing_counts_and_model():
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
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
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
    def ok_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    good = OllamaClient(transport=httpx.MockTransport(ok_handler))
    assert good.health() is True

    def dead_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    dead = OllamaClient(transport=httpx.MockTransport(dead_handler))
    assert dead.health() is False


def test_http_error_raises():
    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = OllamaClient(transport=httpx.MockTransport(fail_handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("system", "user")
