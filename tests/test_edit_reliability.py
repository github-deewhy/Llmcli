"""Edit reliability and stopping for M3 part B (spec 16, section 3): helpful edit errors (B1),
tolerant matching (B2), the line-range edit tool (B3), stop instruction and repeat guard (B4)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from llmcli.agent import REPEAT_NOTE, STOP_INSTRUCTION, CodingAgent
from llmcli.bench.__main__ import main
from llmcli.bench.runner import AgentSettings, run_task
from llmcli.bench.tasks import load_oracle, load_task
from llmcli.coding_tools import TOOL_DEFINITIONS, ProjectCodingTools, tool_schemas
from llmcli.config_file import parse_config_text
from llmcli.edit_matching import MAX_CLOSEST_LINES, closest_lines, tolerant_match
from llmcli.history import Conversation
from llmcli.providers.base import ToolCall
from llmcli.providers.fake import ScriptedProvider, text_turn, tool_turn
from llmcli.ui import StreamingUI
from tests.test_bench import fake_config, make_task

SOURCE = (
    "def greet(name):\n"
    '    return "hello " + name\n'
    "\n"
    "\n"
    "def add(a, b):\n"
    "    return a + b   \n"
    "\n"
    "def shout(text):\n"
    '    return text.upper() + "\\n"\n'
)


def make_tools(tmp_path: Path, text: str = SOURCE, **settings) -> tuple[ProjectCodingTools, Path, StringIO]:
    (tmp_path / "app.py").write_text(text, encoding="utf-8")
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False, width=120))
    return ProjectCodingTools(tmp_path, ui, **settings), tmp_path / "app.py", output


def edit(tools, old, new, approve=True):
    arguments = {"path": "app.py", "old_text": old, "new_text": new}
    return json.loads(tools.execute("replace_in_file", arguments, lambda _m: approve))


# --- B1: helpful error when old_text is not found ---------------------------------------

def test_missing_old_text_shows_closest_numbered_lines(tmp_path):
    tools, path, _ = make_tools(tmp_path)
    result = edit(tools, "def add(a, b):\n    return a - b", "x")
    error = result["error"]
    assert error.startswith("Expected old_text exactly once, found it 0 times.")
    assert "5: def add(a, b):" in error and "6:     return a + b" in error
    assert path.read_text() == SOURCE


def test_closest_lines_are_capped(tmp_path):
    text = "".join(f"line_{n} = {n}\n" for n in range(200))
    shown = closest_lines(text, "line_100 = 999")
    assert shown is not None
    numbered = shown.splitlines()
    assert len(numbered) == MAX_CLOSEST_LINES
    assert "101: line_100 = 100" in numbered


def test_closest_lines_for_empty_file():
    assert closest_lines("", "anything") is None


def test_several_exact_matches_still_refused(tmp_path):
    tools, _, _ = make_tools(tmp_path, "x = 1\nx = 1\n")
    assert "found it 2 times" in edit(tools, "x = 1", "x = 2")["error"]


# --- B2: tolerant matching ---------------------------------------------------------------

def test_exact_mode_does_not_tolerate(tmp_path):
    tools, path, _ = make_tools(tmp_path)
    result = edit(tools, "    return a + b\n", "    return a * b\n")
    assert "found it 0 times" in result["error"]
    assert path.read_text() == SOURCE


def test_tolerant_ignores_trailing_whitespace_and_says_so(tmp_path):
    tools, path, _ = make_tools(tmp_path, edit_matching="tolerant")
    result = edit(tools, "def add(a, b):\n    return a + b\n", "def add(a, b):\n    return a * b\n")
    assert result["status"] == "updated"
    assert "trailing whitespace" in result["note"]
    assert "    return a * b\n" in path.read_text()


def test_tolerant_undoes_doubled_quote_escapes_in_both_texts(tmp_path):
    # The Bolt T03 failure: the model's arguments carry \" where the file has ".
    tools, path, _ = make_tools(tmp_path, edit_matching="tolerant")
    result = edit(tools, '    return \\"hello \\" + name', '    return \\"hi \\" + name')
    assert result["status"] == "updated"
    assert "escaped quotes" in result["note"]
    assert '    return "hi " + name\n' in path.read_text()
    assert '\\"' not in path.read_text()


def test_tolerant_does_not_undo_escaped_newlines(tmp_path):
    tools, path, _ = make_tools(tmp_path, edit_matching="tolerant")
    # The file contains the two characters backslash and n; old_text with a real newline must not match them.
    result = edit(tools, 'text.upper() + "\n"', 'text.lower() + "\n"')
    assert "found it 0 times" in result["error"]
    assert path.read_text() == SOURCE


def test_tolerant_refuses_several_matches(tmp_path):
    text = "a = 1  \nb = 2\na = 1 \nb = 2\n"
    tools, path, _ = make_tools(tmp_path, text, edit_matching="tolerant")
    result = edit(tools, "a = 1\nb = 2", "a = 3\nb = 2")
    assert "2 times with trailing whitespace" in result["error"]
    assert path.read_text() == text


def test_tolerant_with_no_match_gives_closest_lines(tmp_path):
    tools, _, _ = make_tools(tmp_path, edit_matching="tolerant")
    result = edit(tools, "def add(x, y):", "def add(a, b, c):")
    assert "found it 0 times" in result["error"] and "5: def add(a, b):" in result["error"]


def test_tolerant_match_function_counts_places():
    match, places = tolerant_match("x = 1 \n", "x = 1\n", "x = 2\n")
    assert places == 1 and match is not None and match.new_text == "x = 2\n"
    assert tolerant_match("abc\n", "zzz", "y") == (None, 0)


def test_tolerant_edit_still_needs_approval(tmp_path):
    tools, path, _ = make_tools(tmp_path, edit_matching="tolerant")
    result = edit(tools, "    return a + b\n", "    return 0\n", approve=False)
    assert result["status"] == "cancelled"
    assert path.read_text() == SOURCE


# --- B3: edit_lines ---------------------------------------------------------------------

def edit_lines(tools, start, end, new, approve=True):
    arguments = {"path": "app.py", "start_line": start, "end_line": end, "new_text": new}
    return json.loads(tools.execute("edit_lines", arguments, lambda _m: approve))


def test_edit_lines_offered_only_with_setting(tmp_path):
    names = [d["function"]["name"] for d in tool_schemas("replace+lines")]
    assert names.index("edit_lines") == names.index("replace_in_file") + 1
    assert tool_schemas() is TOOL_DEFINITIONS
    tools, _, _ = make_tools(tmp_path)
    assert json.loads(tools.execute("edit_lines", {}, lambda _m: True)) == {"error": "Unknown tool: edit_lines"}


def test_edit_lines_replaces_range_and_shows_diff(tmp_path):
    tools, path, output = make_tools(tmp_path, edit_tools="replace+lines")
    result = edit_lines(tools, 5, 6, "def add(a, b):\n    return a * b")
    assert result["status"] == "updated" and "-    return a + b" in result["diff"]
    lines = path.read_text().splitlines()
    assert lines[4:6] == ["def add(a, b):", "    return a * b"]
    assert len(lines) == len(SOURCE.splitlines())
    assert "return a * b" in output.getvalue()


def test_edit_lines_empty_text_deletes(tmp_path):
    tools, path, _ = make_tools(tmp_path, edit_tools="replace+lines")
    assert edit_lines(tools, 3, 4, "")["status"] == "updated"
    assert path.read_text().splitlines()[2] == "def add(a, b):"


@pytest.mark.parametrize("start, end, fragment", [
    (0, 1, "start_line must be a line number from 1 to 9"),
    (1, 10, "end_line must be a line number from 1 to 9"),
    (5, 4, "end_line must not be smaller than start_line"),
    (True, 2, "start_line must be a line number"),
])
def test_edit_lines_bounds(tmp_path, start, end, fragment):
    tools, path, _ = make_tools(tmp_path, edit_tools="replace+lines")
    assert fragment in edit_lines(tools, start, end, "x")["error"]
    assert path.read_text() == SOURCE


def test_edit_lines_cancelled_leaves_file(tmp_path):
    tools, path, _ = make_tools(tmp_path, edit_tools="replace+lines")
    assert edit_lines(tools, 1, 1, "def hi():", approve=False)["status"] == "cancelled"
    assert path.read_text() == SOURCE


# --- B4: stop instruction and repeat guard ------------------------------------------------

def make_agent(tmp_path, turns, stop_guard):
    tools, _, _ = make_tools(tmp_path)
    ui = tools.ui
    provider = ScriptedProvider(turns)
    agent = CodingAgent(provider, Conversation("ctx", max_tokens=8192), tools, ui, lambda _m: True,
                        "auto", stop_guard)
    return agent, provider


def test_repeated_identical_call_is_skipped_with_note(tmp_path):
    read = ToolCall("c1", "read_file", {"path": "app.py"})
    again = ToolCall("c2", "read_file", {"path": "app.py"})
    agent, _ = make_agent(tmp_path, [tool_turn(read), tool_turn(again), text_turn("done")], True)
    outcome = agent.run("read it")
    assert (outcome.tool_calls, outcome.repeat_skips, outcome.tool_errors) == (2, 1, 0)
    results = [m["content"] for m in agent.conversation.messages if m.get("role") == "tool"]
    assert json.loads(results[1]) == {"status": "skipped", "note": REPEAT_NOTE}
    assert "lines" in json.loads(results[0])


def test_different_calls_and_guard_off_are_not_skipped(tmp_path):
    read = ToolCall("c1", "read_file", {"path": "app.py"})
    other = ToolCall("c2", "read_file", {"path": "app.py", "start_line": 2})
    agent, _ = make_agent(tmp_path, [tool_turn(read), tool_turn(other), text_turn("ok")], True)
    assert agent.run("x").repeat_skips == 0
    agent, _ = make_agent(tmp_path, [tool_turn(read), tool_turn(read), text_turn("ok")], False)
    assert agent.run("x").repeat_skips == 0


def test_stop_instruction_only_with_guard(tmp_path):
    for guard in (True, False):
        agent, provider = make_agent(tmp_path, [text_turn("hi")], guard)
        agent.run("hello")
        messages, _tools = provider.requests[0]
        system = messages[0]["content"]
        assert (STOP_INSTRUCTION in system) is guard


def test_defaults_keep_interactive_behaviour(tmp_path):
    tools, _, _ = make_tools(tmp_path)
    assert (tools.edit_matching, tools.edit_tools) == ("exact", "replace")
    assert tools.schemas() is TOOL_DEFINITIONS


# --- settings: config file and benchmark ------------------------------------------------

def parse(text):
    return parse_config_text(text, Path("/home/u/.llmcli/config.toml"))


def test_settings_are_validated_like_other_keys():
    file = parse('[defaults]\nedit_matching = "tolerant"\nstop_guard = true\n'
                 '[profiles.p]\nprovider = "ollama"\nendpoint = "http://h"\nmodel = "m"\nedit_tools = "replace+lines"\n')
    assert file.defaults["stop_guard"] is True and file.profiles["p"]["edit_tools"] == "replace+lines"
    with pytest.raises(ValueError, match="edit_matching must be one of exact, tolerant"):
        parse('[defaults]\nedit_matching = "fuzzy"\n')
    with pytest.raises(ValueError, match="stop_guard must be true or false"):
        parse('[defaults]\nstop_guard = "yes"\n')


def test_config_carries_settings(monkeypatch):
    from llmcli.config import load_config
    file = parse('[defaults]\nstop_guard = true\n[profiles.p]\nprovider = "ollama"\n'
                 'endpoint = "http://h"\nmodel = "m"\nedit_matching = "tolerant"\n')
    config = load_config({}, file=file)
    assert (config.edit_matching, config.edit_tools, config.stop_guard) == ("tolerant", "replace", True)
    assert load_config({}, file=parse('[profiles.p]\nprovider = "ollama"\nendpoint = "http://h"\nmodel = "m"\n')).stop_guard is True


def test_run_task_uses_settings(tmp_path):
    task = load_task(make_task(tmp_path, oracle=[
        {"tool_calls": [{"name": "edit_lines",
                         "arguments": {"path": "app.py", "start_line": 1, "end_line": 1, "new_text": "VALUE = 2"}}]},
        {"text": "Done."},
    ]))
    record = run_task(task, ScriptedProvider(load_oracle(task)), "p", context_size=8192,
                      settings=AgentSettings(edit_tools="replace+lines"))
    assert record.passed and record.outcome.tool_errors == 0
    record = run_task(task, ScriptedProvider(load_oracle(task)), "p", context_size=8192)
    assert not record.passed and record.outcome.invalid_calls == 1


def test_set_overrides_reach_run_and_report(tmp_path, monkeypatch):
    make_task(tmp_path)
    task = load_task(tmp_path / "tasks" / "X01-sample")
    seen = []

    def fake_run_task(*args, **kwargs):
        seen.append(kwargs["settings"])
        return run_task(task, ScriptedProvider(load_oracle(task)), "p", context_size=8192)

    monkeypatch.setattr("llmcli.config.load_config", lambda overrides, profile: fake_config())
    monkeypatch.setattr("llmcli.providers.create_provider", lambda config: ScriptedProvider([]))
    monkeypatch.setattr("llmcli.bench.__main__.run_task", fake_run_task)
    out = tmp_path / "out"
    code = main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--profile", "p",
                 "--i-understand-this-runs-model-code", "--repeats", "1", "--out", str(out),
                 "--set", "edit_matching=tolerant", "--set", "stop_guard=true"])
    assert code == 0
    assert seen == [AgentSettings("tolerant", "replace", True)]
    assert AgentSettings().stop_guard is True
    report = json.loads((out / "report.json").read_text())
    assert report["environment"]["settings"]["overrides"] == {"edit_matching": "tolerant", "stop_guard": True}
    profile = report["environment"]["profiles"][0]
    assert (profile["edit_matching"], profile["stop_guard"]) == ("tolerant", True)
    md = (out / "report.md").read_text()
    assert "edit_matching tolerant" in md and "edit_matching=tolerant, stop_guard=True" in md


@pytest.mark.parametrize("value", ["color=red", "edit_matching=fuzzy", "stop_guard=maybe", "edit_tools"])
def test_bad_set_values_are_rejected(tmp_path, value, capsys):
    make_task(tmp_path)
    with pytest.raises(SystemExit):
        main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--fake", "--set", value])
    assert "Traceback" not in capsys.readouterr().err
