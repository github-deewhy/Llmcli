"""Agent streaming, tool-call fallback ladder, config keys and /status (spec 14, section 5, cases 11-14)."""

import json
from io import StringIO

import pytest
from rich.console import Console

from llmcli.agent import CodingAgent
from llmcli.cli import handle_command, run_oneshot
from llmcli.coding_tools import ProjectCodingTools
from llmcli.config import Config, load_config
from llmcli.history import Conversation
from llmcli.memory import MemoryStore
from llmcli.providers.base import ToolCall
from llmcli.providers.events import Done, ReasoningDelta, TextDelta, Usage
from llmcli.providers.fake import INTERRUPT, ScriptedProvider, text_turn, tool_turn
from llmcli.ui import StreamingUI

TEXT_CALL = '```json\n{"name": "read_file", "arguments": {"path": "app.py"}}\n```'


def make_ui():
    output = StringIO()
    return StreamingUI(Console(file=output, force_terminal=False, width=120)), output


def make_agent(tmp_path, turns, tool_mode="auto", confirm=lambda _m: False):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    ui, output = make_ui()
    provider = ScriptedProvider(turns)
    conversation = Conversation("Coding context", max_tokens=8192)
    agent = CodingAgent(provider, conversation, ProjectCodingTools(root, ui), ui, confirm, tool_mode)
    return agent, provider, conversation, output


def text_reply(text):
    return [TextDelta(text), Done("stop", {"role": "assistant", "content": text})]


# --- case 11: ladder --------------------------------------------------------------------

def test_text_tool_call_is_executed_and_result_returned_as_user_message(tmp_path):
    agent, provider, conversation, _ = make_agent(
        tmp_path, [text_reply(TEXT_CALL), text_turn("It defines greet.")]
    )
    agent.run("Which function is in app.py?")

    second_request = provider.requests[1][0]
    assert second_request[-1]["role"] == "user"
    assert second_request[-1]["content"].startswith("Tool result for read_file:")
    assert "greet" in second_request[-1]["content"]
    assert agent.mode == "native"  # one text call is not enough to switch


def test_two_text_calls_in_auto_switch_to_text_mode(tmp_path):
    agent, provider, _, output = make_agent(
        tmp_path, [text_reply(TEXT_CALL), text_reply(TEXT_CALL), text_turn("Done.")]
    )
    agent.run("Read app.py twice")

    assert agent.mode == "text"
    assert "switched to text tool calls" in output.getvalue()
    messages, tools = provider.requests[2]
    assert tools is None  # text mode sends no native tool schemas
    assert "reply with only one JSON block" in messages[0]["content"]


def test_native_mode_never_parses_or_switches(tmp_path):
    agent, provider, conversation, output = make_agent(
        tmp_path, [text_reply(TEXT_CALL)], tool_mode="native"
    )
    agent.run("Read app.py")

    assert len(provider.requests) == 1 and agent.mode == "native"
    assert "No tools were invoked" in output.getvalue()


def test_text_tool_mode_starts_in_text_mode(tmp_path):
    agent, provider, _, _ = make_agent(tmp_path, [text_turn("Hello.")], tool_mode="text")
    agent.run("hi")
    assert provider.requests[0][1] is None


def test_bad_text_call_gets_one_repair_then_stops(tmp_path):
    bad = '```json\n{"name": "read_file", "arguments": {}}\n```'
    agent, provider, conversation, output = make_agent(
        tmp_path, [text_reply(bad), text_reply(bad)]
    )
    agent.run("Read app.py")

    assert len(provider.requests) == 2
    repair = provider.requests[1][0][-1]
    assert repair["role"] == "user" and '"path" missing; expected string.' in repair["content"]
    assert "Tool call rejected" in output.getvalue()


def test_native_call_with_bad_arguments_returns_error_to_model(tmp_path):
    bad_call = ToolCall("c1", "read_file", {"start_line": 1})
    agent, provider, conversation, _ = make_agent(
        tmp_path, [tool_turn(bad_call), text_turn("Sorry.")]
    )
    agent.run("Read")
    tool_message = provider.requests[1][0][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"]) == {"error": 'read_file: "path" missing; expected string.'}


# --- case 12: streaming and cancel ---------------------------------------------------------

def test_streamed_text_and_reasoning_are_shown_but_reasoning_not_stored(tmp_path):
    turn = [
        ReasoningDelta("private plan"),
        TextDelta("Hello "),
        TextDelta("there."),
        Usage(prompt_tokens=7, completion_tokens=2, ttft_s=0.5, gen_tokens_per_s=4.0),
        Done("stop", {"role": "assistant", "content": "Hello there."}),
    ]
    agent, provider, conversation, output = make_agent(tmp_path, [turn])
    agent.run("hi")

    assert "Hello there." in output.getvalue()
    assert "private plan" not in output.getvalue()  # hidden by default (P7)
    assert "private plan" not in str(conversation.messages)
    assert provider.last_usage.completion_tokens == 2


def test_reasoning_shown_when_enabled(tmp_path):
    turn = [ReasoningDelta("private plan"), *text_reply("Answer.")]
    agent, _, _, output = make_agent(tmp_path, [turn])
    agent.ui.show_reasoning = True
    agent.run("hi")
    assert "private plan" in output.getvalue()


def test_ctrl_c_mid_stream_discards_partial_reply_and_keeps_history(tmp_path):
    agent, provider, conversation, output = make_agent(
        tmp_path, [[TextDelta("Half an ans"), INTERRUPT]]
    )
    conversation.add("user", "Earlier")
    conversation.add("assistant", "Earlier answer")
    before = list(conversation.messages)

    agent.run("New question")

    assert conversation.messages == before
    assert provider.closed_streams == 1
    assert "Generation cancelled." in output.getvalue()


def test_provider_error_is_one_line(tmp_path):
    from llmcli.providers.events import ProviderTimeout

    agent, _, _, output = make_agent(tmp_path, [[ProviderTimeout("No data from x for 300 s.")]])
    agent.run("hi")
    assert "Agent request failed: No data from x for 300 s." in output.getvalue()
    assert "Traceback" not in output.getvalue()


def test_oneshot_streams_and_stores_answer():
    ui, output = make_ui()
    conversation = Conversation(system_prompt="s", max_tokens=8192)
    config = Config("ollama", "m", "http://localhost:11434", "", 8192, 1024, "AGENTS.md")
    provider = ScriptedProvider([[ReasoningDelta("x"), *text_reply("Final.")]])
    assert run_oneshot(config, provider, conversation, ui, "q") == 0
    assert conversation.messages[-1] == {"role": "assistant", "content": "Final."}


# --- case 13: config keys ---------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".llmcli").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    return home


def write(home, text):
    (home / ".llmcli" / "config.toml").write_text(text)


def test_request_timeout_and_tool_mode_defaults(home):
    config = load_config({})
    assert config.request_timeout == 300 and config.tool_mode == "auto"


def test_profile_overrides_defaults_for_new_keys(home):
    write(home, '[defaults]\nrequest_timeout = 600\ntool_mode = "native"\n'
                '[profiles.a]\nprovider = "ollama"\nendpoint = "http://localhost:11434"\n'
                'model = "m"\ntool_mode = "text"\n')
    config = load_config({})
    assert config.request_timeout == 600 and config.tool_mode == "text"


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("request_timeout = 0", "request_timeout must be a whole number"),
        ('request_timeout = "5m"', "request_timeout must be a whole number"),
        ('tool_mode = "smart"', "tool_mode must be one of auto, native, text"),
    ],
)
def test_new_keys_are_validated(home, line, fragment):
    write(home, f"[defaults]\n{line}\n")
    with pytest.raises(ValueError, match=fragment):
        load_config({})


# --- case 14: /status, /thinking, /probe -----------------------------------------------------

def run_command(command, provider, tmp_path, agent=None):
    ui, output = make_ui()
    config = Config("ollama", "m", "http://localhost:11434", "secret-key", 8192, 1024, "AGENTS.md")
    handle_command(
        command,
        Conversation(system_prompt="", max_tokens=8192),
        config,
        provider,
        ui,
        MemoryStore(root=tmp_path / "h", project_root=tmp_path / "p"),
        agent=agent,
    )
    return ui, output.getvalue()


def test_status_shows_capabilities_and_last_turn(tmp_path):
    provider = ScriptedProvider([])
    provider.capabilities.set("context_length", 40960, "server")
    provider.record_usage(Usage(prompt_tokens=100, completion_tokens=20, ttft_s=1.5, gen_tokens_per_s=6.2))
    _, text = run_command("/status", provider, tmp_path)
    assert "Context length" in text and "40960" in text and "server" in text
    assert "first token 1.5 s" in text and "6.2 tokens/s" in text
    assert "context length of 40960" in text  # differs from context_size 8192
    assert "Request timeout:" in text and "300 s" in text
    assert "secret-key" not in text


def test_status_before_any_turn(tmp_path):
    _, text = run_command("/status", ScriptedProvider([]), tmp_path)
    assert "Last turn: none yet" in text and "unknown" in text


def test_status_shows_current_auto_mode(tmp_path):
    agent, provider, _, _ = make_agent(tmp_path, [])
    agent.mode = "text"
    _, text = run_command("/status", provider, tmp_path, agent=agent)
    assert "auto (now: text)" in text


def test_thinking_command_toggles(tmp_path):
    ui, text = run_command("/thinking on", ScriptedProvider([]), tmp_path)
    assert ui.show_reasoning is True and "shown" in text
    _, text = run_command("/thinking maybe", ScriptedProvider([]), tmp_path)
    assert "Usage: /thinking" in text


def test_probe_command_runs_live_probe(tmp_path):
    provider = ScriptedProvider([tool_turn(ToolCall("p", "ping", {"value": 1}))])
    _, text = run_command("/probe", provider, tmp_path)
    assert provider.capabilities.native_tools == "yes"
    assert "live probe" in text
