"""M3 part C: budgeting correctness (spec 16, section 4: C1 and C2)."""

import json
from io import StringIO

from rich.console import Console

from llmcli.agent import CodingAgent
from llmcli.budget import (
    TokenEstimator,
    TrimResult,
    messages_chars,
    over_budget_warning,
    reply_reserve,
    schemas_chars,
    trim_notice,
)
from llmcli.cli import run_oneshot
from llmcli.coding_tools import ProjectCodingTools
from llmcli.history import Conversation
from llmcli.providers.base import ToolCall
from llmcli.providers.events import Done, TextDelta, ToolCallEvent, Usage
from llmcli.providers.fake import ScriptedProvider
from llmcli.ui import StreamingUI


def make_ui() -> tuple[StreamingUI, StringIO]:
    output = StringIO()
    return StreamingUI(Console(file=output, force_terminal=False, width=400)), output


def answer(text: str, usage: Usage | None = None) -> list:
    events: list = [TextDelta(text)]
    if usage is not None:
        events.append(usage)
    events.append(Done("stop", {"role": "assistant", "content": text}))
    return events


def read_call(call_id: str = "call-1") -> list:
    call = ToolCall(call_id, "read_file", {"path": "app.py"})
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"app.py"}'}}],
    }
    return [ToolCallEvent(call), Done("tool_calls", message)]


def make_agent(tmp_path, turns, conversation):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    ui, output = make_ui()
    provider = ScriptedProvider(turns)
    agent = CodingAgent(provider, conversation, ProjectCodingTools(root, ui), ui, lambda _m: True)
    return agent, provider, output


def old_turns(conversation: Conversation, count: int, size: int) -> None:
    for index in range(count):
        conversation.add("user", f"q{index} " + "x" * size)
        conversation.add("assistant", f"a{index} " + "y" * size)


# --- C1a/C1b: estimator and calibration ----------------------------------------------

def test_estimator_defaults_to_four_characters_per_token():
    estimator = TokenEstimator()
    assert estimator.tokens(400) == 100
    assert estimator.tokens(0) == 0
    assert not estimator.calibrated


def test_calibration_uses_reported_prompt_tokens():
    estimator = TokenEstimator()
    estimator.calibrate(sent_chars=3000, prompt_tokens=1000)
    assert estimator.chars_per_token == 3.0
    assert estimator.tokens(3000) == 1000
    assert estimator.calibrated


def test_calibration_uses_only_the_latest_request():
    estimator = TokenEstimator()
    estimator.calibrate(3000, 1000)
    estimator.calibrate(5000, 1000)
    assert estimator.chars_per_token == 5.0


def test_calibration_is_limited_to_two_to_six_characters_per_token():
    estimator = TokenEstimator()
    estimator.calibrate(1000, 1000)
    assert estimator.chars_per_token == 2.0
    estimator.calibrate(10_000, 1000)
    assert estimator.chars_per_token == 6.0


def test_calibration_ignores_missing_or_zero_usage():
    estimator = TokenEstimator()
    estimator.calibrate(3000, None)
    estimator.calibrate(3000, 0)
    estimator.calibrate(0, 1000)
    assert estimator.chars_per_token == 4.0
    assert not estimator.calibrated


def test_reset_returns_to_the_default():
    estimator = TokenEstimator()
    estimator.calibrate(3000, 1000)
    estimator.reset()
    assert estimator.chars_per_token == 4.0
    assert not estimator.calibrated


# --- Reply reserve -------------------------------------------------------------------

def test_reply_reserve_is_max_tokens_up_to_a_quarter_of_the_window():
    assert reply_reserve(8192, 1024) == 1024
    assert reply_reserve(8192, 4096) == 2048
    assert reply_reserve(4096, 4096) == 1024
    assert reply_reserve(8192, None) == 0
    assert reply_reserve(8192, 0) == 0


def test_prompt_budget_subtracts_the_reply_reserve():
    assert Conversation("", max_tokens=8192, reply_tokens=1024).prompt_budget() == 7168
    assert Conversation("", max_tokens=8192).prompt_budget() == 8192


# --- C1c/C1d: fixed part counted, trimming before the request ------------------------

def test_adding_messages_does_not_trim():
    conversation = Conversation("", max_tokens=10)
    old_turns(conversation, 3, 100)
    assert len(conversation.messages) == 6
    assert conversation.trim_count == 0


def test_fixed_part_counts_towards_the_budget():
    # Messages alone fit (2 turns x ~50 tokens = ~100 of 300); with 250 tokens of
    # fixed part (system prompt and schemas) they do not.
    conversation = Conversation("", max_tokens=300)
    old_turns(conversation, 2, 100)
    assert conversation.fit(0).turns_dropped == 0
    result = conversation.fit(1000)
    assert result.turns_dropped == 1
    assert conversation.messages[0]["content"].startswith("q1")
    assert conversation.trim_count == 1


def test_fit_reports_tokens_freed_and_budget():
    conversation = Conversation("", max_tokens=300, reply_tokens=50)
    old_turns(conversation, 3, 200)
    result = conversation.fit(400)
    assert result.budget == 250
    assert result.turns_dropped == 2
    assert result.tokens_freed == 2 * TokenEstimator().tokens(2 * 203)
    assert not result.over_budget
    assert result.estimated_tokens <= result.budget


def test_latest_turn_is_never_dropped_and_overflow_is_reported():
    conversation = Conversation("", max_tokens=100)
    conversation.add("user", "z" * 2000)
    result = conversation.fit(0)
    assert result.turns_dropped == 0
    assert result.over_budget
    assert len(conversation.messages) == 1


def test_trimming_keeps_tool_calls_with_their_results():
    conversation = Conversation("", max_tokens=60)
    conversation.add("user", "old " + "x" * 100)
    conversation.add_message({"role": "assistant", "content": None,
                              "tool_calls": [{"id": "c1", "function": {"name": "read_file"}}]})
    conversation.add_message({"role": "tool", "tool_call_id": "c1", "content": "y" * 100})
    conversation.add("user", "new")
    conversation.add_message({"role": "assistant", "content": None,
                              "tool_calls": [{"id": "c2", "function": {"name": "read_file"}}]})
    conversation.add_message({"role": "tool", "tool_call_id": "c2", "content": "ok"})
    conversation.fit(0)
    assert [m["role"] for m in conversation.messages] == ["user", "assistant", "tool"]
    assert conversation.messages[2]["tool_call_id"] == "c2"


def test_calibration_changes_what_fits():
    conversation = Conversation("", max_tokens=150)
    old_turns(conversation, 2, 100)  # ~412 characters: 103 tokens at 4 chars/token
    assert conversation.fit(0).turns_dropped == 0
    conversation.estimator.calibrate(2000, 1000)  # 2 chars/token: ~206 tokens
    assert conversation.fit(0).turns_dropped == 1


def test_schemas_chars_counts_only_sent_schemas():
    schemas = [{"type": "function", "function": {"name": "read_file"}}]
    assert schemas_chars(None) == 0
    assert schemas_chars([]) == 0
    assert schemas_chars(schemas) == len(json.dumps(schemas))


def test_agent_counts_system_message_and_schemas(tmp_path, monkeypatch):
    seen: list[int] = []
    original = Conversation.fit

    def spy(self, fixed_chars):
        seen.append(fixed_chars)
        return original(self, fixed_chars)

    monkeypatch.setattr(Conversation, "fit", spy)
    conversation = Conversation("Coding context", max_tokens=8192)
    agent, provider, _ = make_agent(tmp_path, [answer("done")], conversation)
    agent.run("hi")
    messages, schemas = provider.requests[0]
    assert seen == [len(messages[0]["content"]) + schemas_chars(schemas)]
    assert "Coding context" in messages[0]["content"]
    assert schemas


def test_agent_calibrates_from_usage(tmp_path):
    conversation = Conversation("Coding context", max_tokens=8192)
    agent, provider, _ = make_agent(tmp_path, [answer("done", Usage(prompt_tokens=500))], conversation)
    agent.run("hi")
    messages, schemas = provider.requests[0]
    sent = messages_chars(messages) + schemas_chars(schemas)
    assert conversation.estimator.calibrated
    assert conversation.estimator.chars_per_token == min(6.0, max(2.0, sent / 500))


def test_agent_keeps_default_estimate_without_usage(tmp_path):
    conversation = Conversation("Coding context", max_tokens=8192)
    agent, _, _ = make_agent(tmp_path, [answer("done")], conversation)
    agent.run("hi")
    assert conversation.estimator.chars_per_token == 4.0


def test_agent_trims_before_request_and_sends_trimmed_history(tmp_path):
    conversation = Conversation("ctx", max_tokens=3000)
    old_turns(conversation, 3, 2000)
    agent, provider, output = make_agent(tmp_path, [answer("done")], conversation)
    agent.run("new question")
    sent = provider.requests[0][0]
    assert sent[0]["role"] == "system"
    assert not any(str(m.get("content", "")).startswith("q0") for m in sent)
    assert sent[-1]["content"] == "new question"
    assert "History trimmed: dropped" in output.getvalue()
    assert conversation.trim_count >= 1


# --- C2: tell the user ---------------------------------------------------------------

def test_trim_notice_text():
    assert trim_notice(TrimResult(), 8192) is None
    one = trim_notice(TrimResult(turns_dropped=1, tokens_freed=1900), 8192)
    assert one == "History trimmed: dropped 1 oldest turn (~1,900 tokens) to fit context_size 8192."
    two = trim_notice(TrimResult(turns_dropped=2, tokens_freed=50), 4096)
    assert two is not None and "2 oldest turns" in two


def test_over_budget_warning_text():
    assert over_budget_warning(TrimResult(), 8192) is None
    text = over_budget_warning(
        TrimResult(estimated_tokens=9000, budget=7168, over_budget=True), 8192
    )
    assert text is not None
    assert "~9,000 tokens" in text and "7,168" in text and "1,024 kept for the reply" in text
    assert "/clear" in text


def test_no_notice_when_nothing_is_trimmed(tmp_path):
    conversation = Conversation("ctx", max_tokens=8192, reply_tokens=1024)
    agent, _, output = make_agent(tmp_path, [answer("done")], conversation)
    agent.run("hi")
    text = output.getvalue()
    assert "History trimmed" not in text
    assert "Context nearly full" not in text


def test_over_budget_warning_shown_once_per_run(tmp_path):
    # One big tool result fills the window; nothing older can be dropped.
    conversation = Conversation("ctx", max_tokens=200)
    turns = [read_call("c1"), read_call("c2"), answer("done")]
    agent, provider, output = make_agent(tmp_path, turns, conversation)
    agent.run("z" * 2000)
    assert len(provider.requests) == 3
    assert output.getvalue().count("Context nearly full") == 1


def test_oneshot_trims_with_notice_and_calibrates(tmp_path):
    conversation = Conversation("sys", max_tokens=3000)
    old_turns(conversation, 3, 2000)
    ui, output = make_ui()
    provider = ScriptedProvider([answer("hello", Usage(prompt_tokens=400))])

    class Config:
        pass

    code = run_oneshot(Config(), provider, conversation, ui, "hi")
    assert code == 0
    assert "History trimmed: dropped" in output.getvalue()
    assert conversation.estimator.calibrated
    assert provider.requests[0][0][-1]["content"] == "hi"


def test_profile_switch_resets_calibration_and_reserve(monkeypatch):
    from llmcli import cli

    conversation = Conversation("", max_tokens=8192, reply_tokens=1024)
    conversation.estimator.calibrate(3000, 1000)

    class NewConfig:
        provider = "ollama"
        model = "m"
        context_size = 4096
        max_tokens = 512
        tool_mode = "auto"
        edit_matching = "exact"
        edit_tools = "replace"
        stop_guard = True

    class FakeProvider:
        def check_health(self):
            return True, "ok"

        def probe(self):
            return None

        def close(self):
            return None

    class FakeAgent:
        provider = FakeProvider()

        def set_provider(self, *_args):
            return None

        def apply_settings(self, *_args):
            return None

    class Config:
        pass

    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: NewConfig())
    monkeypatch.setattr(cli, "create_provider", lambda _config: FakeProvider())
    ui, _ = make_ui()
    result = cli.switch_profile_session("other", False, Config(), conversation, FakeAgent(), ui)
    assert result is not None
    assert conversation.max_tokens == 4096
    assert conversation.reply_tokens == 512
    assert not conversation.estimator.calibrated


# --- 0.8.1: text-mode tool results do not start a turn -------------------------------

TEXT_CALL = '```json\n{"name": "read_file", "arguments": {"path": "app.py"}}\n```'


def text_mode_turn(conversation: Conversation, prompt: str, results: int, size: int) -> None:
    """The message sequence the agent stores for one text-mode task."""
    conversation.add("user", prompt)
    for index in range(results):
        conversation.add("assistant", TEXT_CALL)
        conversation.add("user", f"Tool result for read_file: {index} " + "r" * size, continues_turn=True)


def test_trimming_never_drops_the_task_prompt_inside_a_text_mode_turn():
    conversation = Conversation("", max_tokens=300)
    conversation.add("user", "old " + "x" * 800)
    conversation.add("assistant", "old answer")
    text_mode_turn(conversation, "TASK: add a function", results=3, size=400)
    result = conversation.fit(0)
    assert result.turns_dropped == 1
    assert conversation.messages[0]["content"] == "TASK: add a function"
    assert result.over_budget  # the current turn alone is too big: warned, not cut


def test_repair_notes_continue_the_turn():
    conversation = Conversation("", max_tokens=50)
    conversation.add("user", "TASK")
    conversation.add("assistant", "bad call")
    conversation.add("user", "Tool call error: ... " + "e" * 400, continues_turn=True)
    conversation.fit(0)
    assert conversation.messages[0]["content"] == "TASK"


def test_keep_text_only_keeps_continuation_marks():
    conversation = Conversation("", max_tokens=400)
    text_mode_turn(conversation, "TASK", results=2, size=10)
    conversation.keep_text_only()
    starts = [m["content"] for m in conversation.messages if conversation._starts_turn(m)]
    assert starts == ["TASK"]


def test_clear_forgets_continuation_marks():
    conversation = Conversation("", max_tokens=400)
    text_mode_turn(conversation, "TASK", results=1, size=10)
    conversation.clear()
    assert conversation._continuations == set()


def test_agent_marks_text_mode_tool_results_and_repairs(tmp_path):
    bad = '```json\n{"name": "no_such_tool", "arguments": {}}\n```'
    turns = [answer(TEXT_CALL), answer(bad), answer("done")]
    conversation = Conversation("ctx", max_tokens=8192)
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    ui, _ = make_ui()
    provider = ScriptedProvider(turns)
    agent = CodingAgent(provider, conversation, ProjectCodingTools(root, ui), ui, lambda _m: True, "text")
    agent.run("TASK")
    users = [m for m in conversation.messages if m["role"] == "user"]
    assert len(users) == 3  # task, tool result, repair note
    assert [conversation._starts_turn(m) for m in users] == [True, False, False]
