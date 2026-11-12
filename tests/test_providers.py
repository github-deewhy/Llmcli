"""Tests for structured tool-call provider responses."""

import json

from llmcli.config import Config
from llmcli.providers.base import parse_tool_calls
from llmcli.providers.ollama import OllamaProvider
from llmcli.providers.openai_compat import OpenAICompatProvider


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def make_config(provider):
    return Config(
        provider=provider,
        model="test-model",
        endpoint="http://localhost",
        api_key="test-key",
        context_size=8192,
        max_tokens=1024,
        context_file="CONTEXT.md",
    )


def test_parse_tool_calls_accepts_openai_and_ollama_arguments():
    calls = parse_tool_calls(
        [
            {
                "id": "call-1",
                "function": {"name": "read_file", "arguments": '{"path":"src/main.py"}'},
            },
            {"function": {"name": "list_files", "arguments": {"path": "src"}}},
        ]
    )

    assert calls[0].id == "call-1"
    assert calls[0].arguments == {"path": "src/main.py"}
    assert calls[1].name == "list_files"
    assert calls[1].arguments == {"path": "src"}


def test_parse_tool_calls_rejects_malformed_arguments():
    try:
        parse_tool_calls([{"function": {"name": "read_file", "arguments": "not-json"}}])
    except ValueError as error:
        assert "Invalid arguments" in str(error)
    else:
        raise AssertionError("Expected malformed tool arguments to be rejected")


def test_ollama_complete_parses_native_tool_calls(monkeypatch):
    response = FakeResponse(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "app.py"}}}
                ],
            }
        }
    )
    monkeypatch.setattr("llmcli.providers.ollama.httpx.post", lambda *args, **kwargs: response)

    result = OllamaProvider(make_config("ollama")).complete([], [{"type": "function"}])

    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "app.py"}
    assert result.assistant_message["role"] == "assistant"
    assert OllamaProvider(make_config("ollama")).tool_result_message(
        result.tool_calls[0], "contents"
    ) == {"role": "tool", "tool_name": "read_file", "content": "contents"}


def test_openai_compatible_complete_parses_tool_calls_and_sends_tools(monkeypatch):
    response = FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "app.py"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    request = {}

    def fake_post(url, **kwargs):
        request.update(url=url, **kwargs)
        return response

    monkeypatch.setattr("llmcli.providers.openai_compat.httpx.post", fake_post)
    result = OpenAICompatProvider(make_config("openai_compat")).complete([], [{"type": "function"}])

    assert request["json"]["tools"] == [{"type": "function"}]
    assert request["json"]["tool_choice"] == "auto"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert result.tool_calls[0].arguments == {"path": "app.py"}
