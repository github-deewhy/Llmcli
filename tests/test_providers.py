"""Provider streaming, errors and probing with recorded byte streams (spec 14, section 5, cases 1-9)."""

import json

import httpx
import pytest

from llmcli.config import Config
from llmcli.providers.base import parse_tool_calls
from llmcli.providers.events import (
    Done,
    ProviderConnectError,
    ProviderHTTPError,
    ProviderProtocolError,
    ProviderTimeout,
    ReasoningDelta,
    TextDelta,
    ToolCallEvent,
    Usage,
)
from llmcli.providers.http import make_client
from llmcli.providers.ollama import OllamaProvider
from llmcli.providers.openai_compat import OpenAICompatProvider

SECRET = "sk-test-secret"


def make_config(provider, endpoint="http://localhost:11434", **extra):
    config = Config(
        provider=provider,
        model="test-model",
        endpoint=endpoint,
        api_key=SECRET,
        context_size=8192,
        max_tokens=1024,
        context_file="AGENTS.md",
    )
    for key, value in extra.items():
        setattr(config, key, value)
    return config


def with_transport(provider, handler):
    provider.client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def ndjson(*lines):
    return "".join(json.dumps(line) + "\n" for line in lines)


def sse(*chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def kinds(events):
    return [type(e).__name__ for e in events]


# --- parsing helper (kept from v0.3.0) ---------------------------------------------

def test_parse_tool_calls_accepts_openai_and_ollama_arguments():
    calls = parse_tool_calls(
        [
            {"id": "call-1", "function": {"name": "read_file", "arguments": '{"path":"src/main.py"}'}},
            {"function": {"name": "list_files", "arguments": {"path": "src"}}},
        ]
    )
    assert calls[0].id == "call-1"
    assert calls[0].arguments == {"path": "src/main.py"}
    assert calls[1].arguments == {"path": "src"}


def test_parse_tool_calls_rejects_malformed_arguments():
    with pytest.raises(ValueError, match="Invalid arguments"):
        parse_tool_calls([{"function": {"name": "read_file", "arguments": "not-json"}}])


# --- 1. Ollama stream -------------------------------------------------------------

def test_ollama_stream_text_thinking_tool_calls_and_usage():
    sent = {}

    def handler(request):
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            text=ndjson(
                {"message": {"role": "assistant", "content": "", "thinking": "Let me look."}},
                {"message": {"role": "assistant", "content": "Reading "}},
                {"message": {"role": "assistant", "content": "the file.", "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "app.py"}}}
                ]}},
                {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                 "prompt_eval_count": 120, "eval_count": 20, "eval_duration": 2_000_000_000},
            ),
        )

    provider = with_transport(
        OllamaProvider(make_config("ollama", extra_body={"think": False})), handler
    )
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    events = list(provider.stream([{"role": "user", "content": "hi"}], tools))

    assert kinds(events) == ["ReasoningDelta", "TextDelta", "TextDelta", "ToolCallEvent", "Usage", "Done"]
    assert sent["body"]["stream"] is True and sent["body"]["tools"] == tools
    assert sent["body"]["think"] is False  # extra_body now reaches Ollama
    call = events[3].call
    assert call.name == "read_file" and call.arguments == {"path": "app.py"}
    usage = events[4]
    assert usage.prompt_tokens == 120 and usage.completion_tokens == 20
    assert usage.gen_tokens_per_s == 10.0 and usage.ttft_s is not None
    done = events[-1]
    assert done.assistant_message == {
        "role": "assistant",
        "content": "Reading the file.",
        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "app.py"}}}],
    }
    assert "thinking" not in json.dumps(done.assistant_message)
    assert provider.last_usage == usage
    assert provider.capabilities.native_tools == "yes"
    assert provider.capabilities.reasoning == "field"


# --- 2. OpenAI-compatible tool-call fragments -----------------------------------------

def test_openai_stream_merges_tool_call_fragments_by_index():
    sent = {}

    def handler(request):
        sent["body"] = json.loads(request.content)
        sent["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            text=sse(
                {"choices": [{"delta": {"role": "assistant", "content": "Checking"}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_a", "type": "function",
                     "function": {"name": "read_file", "arguments": ""}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '{"path": "ap'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 1, "id": "call_b", "function": {"name": "list_files", "arguments": "{}"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'p.py"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                 "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
            ),
        )

    provider = with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h/v1")), handler)
    events = list(provider.stream([{"role": "user", "content": "x"}], [{"type": "function"}]))

    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_a", "read_file", {"path": "app.py"}),
        ("call_b", "list_files", {}),
    ]
    assert sent["body"]["stream"] is True and sent["body"]["tool_choice"] == "auto"
    assert sent["auth"] == f"Bearer {SECRET}"
    done = events[-1]
    assert isinstance(done, Done) and done.finish_reason == "tool_calls"
    assert done.assistant_message["content"] == "Checking"
    assert done.assistant_message["tool_calls"][0]["function"]["arguments"] == '{"path": "app.py"}'
    usage = next(e for e in events if isinstance(e, Usage))
    assert usage.prompt_tokens == 50 and usage.completion_tokens == 9
    assert provider.capabilities.streams_tool_calls == "yes"


# --- 3. reasoning_content ---------------------------------------------------------

def test_openai_reasoning_content_is_kept_out_of_history():
    def handler(request):
        return httpx.Response(200, text=sse(
            {"choices": [{"delta": {"reasoning_content": "secret plan"}}]},
            {"choices": [{"delta": {"content": "Answer."}, "finish_reason": "stop"}]},
        ))

    provider = with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h/v1")), handler)
    events = list(provider.stream([{"role": "user", "content": "x"}]))
    assert ReasoningDelta("secret plan") in events
    assert events[-1].assistant_message == {"role": "assistant", "content": "Answer."}


# --- 4. inline <think> split across chunks -----------------------------------------

def test_inline_think_tags_split_across_chunks():
    pieces = ["<th", "ink>step ", "one</th", "ink>Final", " answer"]

    def handler(request):
        return httpx.Response(200, text=sse(*({"choices": [{"delta": {"content": p}}]} for p in pieces)))

    provider = with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h/v1")), handler)
    events = list(provider.stream([{"role": "user", "content": "x"}]))
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert reasoning == "step one" and text == "Final answer"
    assert events[-1].assistant_message["content"] == "Final answer"
    assert provider.capabilities.reasoning == "inline"


# --- 5. malformed streams ------------------------------------------------------------

@pytest.mark.parametrize(
    "body",
    [
        "data: {not json\n\n",
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "read_file", "arguments": "{bad"}}]}}]}),
        sse({"error": {"message": "context overflow"}}),
    ],
)
def test_malformed_openai_stream_raises_protocol_error(body):
    provider = with_transport(
        OpenAICompatProvider(make_config("openai_compat", "http://h/v1")),
        lambda request: httpx.Response(200, text=body),
    )
    with pytest.raises(ProviderProtocolError) as error:
        list(provider.stream([{"role": "user", "content": "x"}]))
    assert "\n" not in str(error.value)


def test_malformed_ollama_line_raises_protocol_error():
    provider = with_transport(
        OllamaProvider(make_config("ollama")),
        lambda request: httpx.Response(200, text="{oops\n"),
    )
    with pytest.raises(ProviderProtocolError, match="could not read"):
        list(provider.stream([{"role": "user", "content": "x"}]))


# --- 6. connection, timeout and HTTP errors -------------------------------------------

def raising(exc):
    def handler(request):
        raise exc("boom", request=request)
    return handler


@pytest.mark.parametrize(
    "handler, error_type, fragment",
    [
        (raising(httpx.ConnectError), ProviderConnectError, "Cannot reach"),
        (raising(httpx.ReadTimeout), ProviderTimeout, "Raise request_timeout"),
        (lambda r: httpx.Response(401, text=f"bad key {SECRET}"), ProviderHTTPError, "HTTP 401"),
        (lambda r: httpx.Response(404, text="model 'x' not found, prompt: hello"), ProviderHTTPError, "model name"),
        (lambda r: httpx.Response(500, text="crash"), ProviderHTTPError, "HTTP 500"),
    ],
)
@pytest.mark.parametrize("kind", ["ollama", "openai_compat"])
def test_errors_are_one_line_without_secrets_or_server_body(kind, handler, error_type, fragment):
    cls = OllamaProvider if kind == "ollama" else OpenAICompatProvider
    provider = with_transport(cls(make_config(kind, "http://h:1/v1")), handler)
    with pytest.raises(error_type) as error:
        list(provider.stream([{"role": "user", "content": "x"}]))
    message = str(error.value)
    assert fragment in message
    assert "\n" not in message and SECRET not in message
    assert "prompt" not in message and "crash" not in message and "Traceback" not in message


def test_timeout_message_names_the_configured_seconds():
    provider = with_transport(
        OllamaProvider(make_config("ollama", request_timeout=42)), raising(httpx.ReadTimeout)
    )
    with pytest.raises(ProviderTimeout, match="for 42 s"):
        list(provider.stream([{"role": "user", "content": "x"}]))


# --- 7. finite timeouts, no environment proxies (P8) ---------------------------------

def test_client_has_finite_timeouts_and_ignores_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")
    client = make_client(77)
    assert client.trust_env is False
    assert client.timeout.read == 77 and client.timeout.connect == 5
    provider = OllamaProvider(make_config("ollama", request_timeout=120))
    assert provider.client.trust_env is False and provider.client.timeout.read == 120


# --- 8. probes --------------------------------------------------------------------------

def test_ollama_probe_reads_capabilities_and_context_length():
    def handler(request):
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960},
        })

    caps = with_transport(OllamaProvider(make_config("ollama")), handler).probe()
    assert (caps.native_tools, caps.reasoning, caps.context_length) == ("yes", "field", 40960)
    assert caps.sources["context_length"] == "server"


def test_llamacpp_probe_reads_props():
    def handler(request):
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": 16384}, "total_slots": 2,
            })
        return httpx.Response(200, json={"data": [{"id": "test-model"}]})

    caps = with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h:9931/v1")), handler).probe()
    assert caps.context_length == 16384 and caps.parallel_slots == 2


@pytest.mark.parametrize("kind", ["ollama", "openai_compat"])
def test_failing_probe_leaves_unknown_and_does_not_raise(kind):
    cls = OllamaProvider if kind == "ollama" else OpenAICompatProvider
    caps = with_transport(cls(make_config(kind, "http://h/v1")), raising(httpx.ConnectError)).probe()
    assert caps.native_tools == "unknown" and caps.context_length == "unknown"


# --- 9. observed capabilities and live probe ------------------------------------------

def test_live_probe_records_tool_support():
    def handler(request):
        return httpx.Response(200, text=ndjson(
            {"message": {"content": "", "tool_calls": [{"function": {"name": "ping", "arguments": {"value": 1}}}]}},
            {"message": {"content": ""}, "done": True, "eval_count": 3},
        ))

    provider = with_transport(OllamaProvider(make_config("ollama")), handler)
    caps = provider.live_probe()
    assert caps.native_tools == "yes" and caps.sources["native_tools"] == "live probe"
    assert caps.usage_reported == "yes"


def test_live_probe_records_missing_tool_support():
    def handler(request):
        return httpx.Response(200, text=sse({"choices": [{"delta": {"content": "pong"}}]}))

    caps = with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h/v1")), handler).live_probe()
    assert caps.native_tools == "no"


# --- existing behaviour kept ----------------------------------------------------------------

def test_ollama_tool_result_message_uses_tool_name():
    provider = OllamaProvider(make_config("ollama"))
    call = parse_tool_calls([{"function": {"name": "read_file", "arguments": {}}}])[0]
    assert provider.tool_result_message(call, "contents") == {
        "role": "tool", "tool_name": "read_file", "content": "contents",
    }


def test_health_check_reports_unreachable_server_without_raising():
    provider = with_transport(OllamaProvider(make_config("ollama")), raising(httpx.ConnectError))
    healthy, message = provider.check_health()
    assert healthy is False and "Cannot reach" in message
