"""Benchmark harness (spec 15, section 4). No network, no model."""

from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from llmcli.agent import MAX_TOOL_ROUNDS, CodingAgent
from llmcli.bench import DEFAULT_TASKS_DIR
from llmcli.bench.__main__ import ACK_FLAG, check_tasks, main
from llmcli.bench.report import build_report, run_metrics, write_report
from llmcli.bench.runner import ANSWER_FILE, checker_environment, run_task
from llmcli.bench.tasks import TaskError, load_oracle, load_task, load_tasks, null_script
from llmcli.coding_tools import ProjectCodingTools
from llmcli.config import Config
from llmcli.history import Conversation
from llmcli.providers.base import ToolCall
from llmcli.providers.events import Done, ProviderConnectError, TextDelta
from llmcli.providers.fake import ScriptedProvider, text_turn, tool_turn
from llmcli.ui import StreamingUI

TEXT_CALL = '```json\n{"name": "read_file", "arguments": {"path": "app.py"}}\n```'
BAD_TEXT_CALL = '```json\n{"name": "read_file", "arguments": {}}\n```'


def make_task(tmp_path: Path, *, checker=None, toml_extra: str = "", oracle=None, check_files=None) -> Path:
    """A minimal task folder: one file in repo/, a checker that runs a script from check/."""
    task = tmp_path / "tasks" / "X01-sample"
    (task / "repo").mkdir(parents=True)
    (task / "check").mkdir()
    (task / "repo" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name, text in (check_files or {"check_value.py": CHECK_VALUE}).items():
        (task / "check" / name).write_text(text, encoding="utf-8")
    checker = checker or ["python", "check_value.py"]
    (task / "task.toml").write_text(
        f'id = "X01"\ntitle = "Sample"\ncategory = "fix-bug"\nprompt = "Set VALUE to 2."\n'
        f"checker = {json.dumps(checker)}\n{toml_extra}",
        encoding="utf-8",
    )
    oracle = oracle or [
        {"tool_calls": [{"name": "replace_in_file",
                         "arguments": {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}}]},
        {"text": "Done, VALUE is 2."},
    ]
    (task / "oracle.json").write_text(json.dumps(oracle), encoding="utf-8")
    return task


CHECK_VALUE = (
    "import sys\nfrom pathlib import Path\n"
    "sys.exit(0 if 'VALUE = 2' in Path('app.py').read_text() else 1)\n"
)


def make_agent(tmp_path, turns, tool_mode="auto"):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    ui = StreamingUI(Console(file=StringIO(), force_terminal=False, width=120))
    provider = ScriptedProvider(turns)
    conversation = Conversation("ctx", max_tokens=8192)
    agent = CodingAgent(provider, conversation, ProjectCodingTools(root, ui), ui, lambda _m: True, tool_mode)
    return agent, provider


def text_reply(text):
    return [TextDelta(text), Done("stop", {"role": "assistant", "content": text})]


# --- 1. task loader -------------------------------------------------------------------

def test_load_valid_task(tmp_path):
    task = load_task(make_task(tmp_path, toml_extra="checker_timeout = 5\n"))
    assert (task.id, task.category, task.checker_timeout) == ("X01", "fix-bug", 5)
    assert task.checker == ("python", "check_value.py")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: (p / "task.toml").write_text('id = "X01"\n'), "missing title, category, prompt, checker"),
        (lambda p: (p / "task.toml").write_text(
            (p / "task.toml").read_text().replace('"fix-bug"', '"poetry"')), "unknown category 'poetry'"),
        (lambda p: __import__("shutil").rmtree(p / "repo"), "the repo/ folder is missing"),
        (lambda p: (p / "task.toml").write_text("id = ["), "not valid TOML"),
        (lambda p: (p / "task.toml").write_text(
            (p / "task.toml").read_text() + "checker_timeout = 0\n"), "checker_timeout must be a positive"),
    ],
)
def test_bad_task_gives_one_line_error(tmp_path, mutate, message):
    path = make_task(tmp_path)
    mutate(path)
    with pytest.raises(TaskError) as error:
        load_task(path)
    assert message in str(error.value)
    assert "\n" not in str(error.value)


def test_unknown_task_id_and_missing_folder(tmp_path):
    make_task(tmp_path)
    with pytest.raises(TaskError, match="Unknown task id: Z9"):
        load_tasks(tmp_path / "tasks", ["Z9"])
    with pytest.raises(TaskError, match="Task folder not found"):
        load_tasks(tmp_path / "nowhere")


def test_oracle_script_becomes_scripted_turns(tmp_path):
    turns = load_oracle(load_task(make_task(tmp_path)))
    assert len(turns) == 2
    assert turns[0][-1].finish_reason == "tool_calls"


# --- 2. every shipped task: oracle passes, null fails ---------------------------------

SHIPPED = load_tasks(DEFAULT_TASKS_DIR)


@pytest.mark.parametrize("task", SHIPPED, ids=[t.id for t in SHIPPED])
def test_shipped_task_oracle_passes_and_null_fails(task):
    [(_, oracle_passed, null_failed)] = check_tasks([task])
    assert oracle_passed, f"{task.id}: oracle does not pass"
    assert null_failed, f"{task.id}: an untouched repo passes the checker"


def test_six_tasks_cover_six_categories():
    assert [t.id for t in SHIPPED] == ["T01", "T02", "T03", "T04", "T05", "T06"]
    assert len({t.category for t in SHIPPED}) == 6


# --- 3. hidden checks overwrite model files -------------------------------------------

def test_hidden_check_overwrites_file_the_model_wrote(tmp_path):
    cheat = "import sys\nsys.exit(0)\n"
    task = load_task(make_task(tmp_path, oracle=[
        {"tool_calls": [{"name": "create_file", "arguments": {"path": "check_value.py", "content": cheat}}]},
        {"text": "Done."},
    ]))
    record = run_task(task, ScriptedProvider(load_oracle(task)), "fake", context_size=8192)
    assert record.outcome.tool_calls == 1
    assert not record.passed  # the real check_value.py replaced the model's version


# --- 4. checker time limit and environment --------------------------------------------

def test_checker_timeout_fails_the_run(tmp_path):
    slow = "import time\ntime.sleep(10)\n"
    task = load_task(make_task(tmp_path, toml_extra="checker_timeout = 1\n",
                               check_files={"check_value.py": slow}))
    record = run_task(task, ScriptedProvider(null_script()), "fake", context_size=8192)
    assert not record.passed and record.check.timed_out
    assert "timed out after 1 s" in record.check.output


def test_checker_sees_only_four_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "abc")
    dump = "import json, os\nfrom pathlib import Path\nPath('env.json').write_text(json.dumps(dict(os.environ)))\n"
    task = load_task(make_task(tmp_path, check_files={"check_value.py": dump}))
    seen = {}

    def capture(task_, work, answer, home):
        result = original(task_, work, answer, home)
        seen.update(json.loads((work / "env.json").read_text()))
        return result

    from llmcli.bench import runner
    original = runner.run_checker
    monkeypatch.setattr(runner, "run_checker", capture)
    run_task(task, ScriptedProvider(null_script()), "fake", context_size=8192)
    # macOS may add __CF_USER_TEXT_ENCODING to a child process; nothing else is allowed.
    assert set(seen) - {"__CF_USER_TEXT_ENCODING"} == {"PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE"}
    assert set(checker_environment(tmp_path)) == {"PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE"}


def test_final_answer_is_written_for_explain_checkers(tmp_path):
    reader = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.exit(0 if 'forty-two' in Path('{ANSWER_FILE}').read_text() else 1)\n"
    )
    task = load_task(make_task(tmp_path, check_files={"check_value.py": reader},
                               oracle=[{"text": "The answer is forty-two."}]))
    assert run_task(task, ScriptedProvider(load_oracle(task)), "fake", context_size=8192).passed


# --- 5. deadline ----------------------------------------------------------------------

class SlowProvider(ScriptedProvider):
    def stream(self, messages, tools=None):
        for event in super().stream(messages, tools):
            time.sleep(0.05)
            yield event


def test_deadline_stops_the_run_and_is_recorded(tmp_path):
    task = load_task(make_task(tmp_path))
    record = run_task(task, SlowProvider([text_turn("thinking " * 50)]), "fake",
                      context_size=8192, deadline_s=0.01)
    assert record.outcome.stop_reason == "provider_error"
    assert record.outcome.error_type == "BenchDeadline"
    assert "deadline of 0.01 s" in (record.outcome.error or "")
    assert not record.passed


# --- 6. RunOutcome --------------------------------------------------------------------

def test_outcome_no_tools(tmp_path):
    agent, _ = make_agent(tmp_path, [text_turn("Hello.")])
    outcome = agent.run("hi")
    assert (outcome.stop_reason, outcome.rounds, outcome.tool_calls) == ("no_tools", 1, 0)
    assert outcome.final_text == "Hello." and len(outcome.usages) == 1


def test_outcome_answered_after_tool(tmp_path):
    call = ToolCall("c1", "read_file", {"path": "app.py"})
    agent, _ = make_agent(tmp_path, [tool_turn(call), text_turn("It defines greet.")])
    outcome = agent.run("what is in app.py?")
    assert (outcome.stop_reason, outcome.rounds, outcome.tool_calls, outcome.invalid_calls) == (
        "answered", 2, 1, 0)


def test_outcome_round_limit(tmp_path):
    call = ToolCall("c1", "list_files", {})
    agent, _ = make_agent(tmp_path, [tool_turn(call) for _ in range(MAX_TOOL_ROUNDS)])
    outcome = agent.run("loop")
    assert (outcome.stop_reason, outcome.rounds) == ("round_limit", MAX_TOOL_ROUNDS)


def test_outcome_invalid_native_call_and_unknown_tool(tmp_path):
    bad = ToolCall("c1", "read_file", {})
    unknown = ToolCall("c2", "delete_everything", {})
    agent, _ = make_agent(tmp_path, [tool_turn(bad, unknown), text_turn("ok")])
    outcome = agent.run("go")
    assert (outcome.tool_calls, outcome.invalid_calls) == (2, 2)


def test_outcome_text_call_repair(tmp_path):
    agent, _ = make_agent(tmp_path, [text_reply(BAD_TEXT_CALL), text_reply(TEXT_CALL), text_turn("greet")])
    outcome = agent.run("read app.py")
    assert (outcome.invalid_calls, outcome.repairs, outcome.tool_calls, outcome.stop_reason) == (
        1, 1, 1, "answered")


def test_outcome_mode_switch(tmp_path):
    agent, _ = make_agent(tmp_path, [text_reply(TEXT_CALL), text_reply(TEXT_CALL), text_turn("Done.")])
    outcome = agent.run("read twice")
    assert outcome.mode_switched and outcome.tool_calls == 2


def test_outcome_provider_error(tmp_path):
    agent, _ = make_agent(tmp_path, [[ProviderConnectError("Cannot reach fake://. Is the server running?")]])
    outcome = agent.run("hi")
    assert outcome.stop_reason == "provider_error"
    assert outcome.error_type == "ProviderConnectError"


def test_conversation_counts_trims():
    conversation = Conversation("", max_tokens=10)
    conversation.add("user", "a" * 40)
    conversation.add("assistant", "b" * 40)
    conversation.add("user", "c" * 40)
    assert conversation.trim_count >= 1


# --- 7. --max-tokens override, 10. acknowledgement flag ------------------------------

def fake_config(**overrides):
    values = dict(provider="openai_compat", model="m", endpoint="http://127.0.0.1:1", api_key="sk-SECRET",
                  context_size=4096, max_tokens=4096, context_file="AGENTS.md", profile="p")
    values.update(overrides)
    return Config(**values)


def test_real_profile_requires_acknowledgement(tmp_path, capsys, monkeypatch):
    make_task(tmp_path)
    called = []
    monkeypatch.setattr("llmcli.config.load_config", lambda *a, **k: called.append(1))
    code = main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--profile", "p", "--out", str(tmp_path / "o")])
    assert code == 2 and not called
    assert ACK_FLAG in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_max_tokens_override_reaches_provider_and_report(tmp_path, monkeypatch):
    make_task(tmp_path)
    task = load_task(tmp_path / "tasks" / "X01-sample")
    built = []

    def create(config):
        built.append(config)
        provider = ScriptedProvider([turn for _ in range(2) for turn in load_oracle(task)])
        return provider

    monkeypatch.setattr("llmcli.config.load_config", lambda overrides, profile: fake_config())
    monkeypatch.setattr("llmcli.providers.create_provider", create)
    out = tmp_path / "out"
    code = main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--profile", "p", ACK_FLAG,
                 "--max-tokens", "333", "--repeats", "2", "--out", str(out)])
    assert code == 0
    assert built[0].max_tokens == 333
    report = json.loads((out / "report.json").read_text())
    assert report["environment"]["settings"]["max_tokens"] == 333
    assert report["environment"]["profiles"][0]["max_tokens"] == 333
    # 8. no api_key value anywhere in the output
    for file in out.rglob("*"):
        if file.is_file():
            assert "sk-SECRET" not in file.read_text()


# --- 8, 9. report and repeats ---------------------------------------------------------

def test_fake_run_writes_report_transcripts_and_aggregates(tmp_path):
    make_task(tmp_path)
    out = tmp_path / "out"
    code = main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--fake", "--repeats", "3", "--out", str(out)])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["schema_version"] == 3
    assert report["configuration"] == "m3b-v0.7"
    assert {"machine", "profiles", "settings", "tasks"} <= set(report["environment"])
    assert len(report["runs"]) == 3
    [cell] = report["cells"]
    assert (cell["runs"], cell["passed"], cell["stop_reasons"]) == (3, 3, {"answered": 3})
    assert set(cell["wall_s"]) == {"mean", "min", "max"}
    assert len(list((out / "transcripts").glob("*.json"))) == 3
    md = (out / "report.md").read_text()
    assert "| X01 | 3/3 |" in md and "## Environment" in md


def test_run_metrics_cache_and_context(tmp_path):
    task = load_task(make_task(tmp_path))
    record = run_task(task, ScriptedProvider(load_oracle(task)), "fake", context_size=100)
    metrics = run_metrics(record)
    assert metrics["peak_prompt_tokens"] == 10
    assert metrics["peak_context_utilisation"] == 0.1
    assert metrics["cache_hit_rate"] is None  # fake provider reports no cached tokens
    assert metrics["tool_call_validity"] == 1.0
    from llmcli.bench.environment import machine_info
    environment = {"machine": machine_info(), "profiles": [],
                   "settings": {"repeats": 1, "deadline_s": 600.0, "max_tokens": None}, "tasks": []}
    report = build_report(environment, [record], "t", "c")
    write_report(tmp_path / "r", report, [record])
    assert (tmp_path / "r" / "transcripts" / "X01-fake-1.json").is_file()


def test_list_and_check_commands(tmp_path, capsys):
    make_task(tmp_path)
    assert main(["--tasks-dir", str(tmp_path / "tasks"), "list"]) == 0
    assert "X01" in capsys.readouterr().out
    assert main(["--tasks-dir", str(tmp_path / "tasks"), "check"]) == 0
    assert "All tasks ok." in capsys.readouterr().out


def test_check_reports_a_broken_task(tmp_path, capsys):
    make_task(tmp_path, oracle=[{"text": "I did nothing."}])
    assert main(["--tasks-dir", str(tmp_path / "tasks"), "check"]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_missing_tasks_dir_is_one_line(tmp_path, capsys):
    assert main(["--tasks-dir", str(tmp_path / "none"), "list"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("Error: Task folder not found") and err.count("\n") == 1


def test_python_in_checker_means_current_interpreter(tmp_path):
    from llmcli.bench.runner import checker_argv
    task = load_task(make_task(tmp_path))
    assert checker_argv(task)[0] == sys.executable
