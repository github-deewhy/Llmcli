"""Tests for approval-gated project tools."""

import json
import subprocess
from io import StringIO

from rich.console import Console

from llmcli.coding_tools import ProjectCodingTools
from llmcli.ui import StreamingUI


def make_tools(tmp_path, output=None):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    console = Console(file=output or StringIO(), force_terminal=False)
    return root, ProjectCodingTools(root, StreamingUI(console))


def test_read_and_search_are_project_scoped_and_block_environment_files(tmp_path):
    root, tools = make_tools(tmp_path)
    (root / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    def approve(_message):
        return True

    read = json.loads(tools.execute("read_file", {"path": "app.py"}, approve))
    blocked_env = json.loads(tools.execute("read_file", {"path": ".env"}, approve))
    blocked_parent = json.loads(tools.execute("read_file", {"path": "../outside.txt"}, approve))
    found = json.loads(tools.execute("search_files", {"query": "hello"}, approve))

    assert "return 'hello'" in read["lines"]
    assert "error" in blocked_env
    assert "error" in blocked_parent
    assert found["matches"] == ["app.py:2:     return 'hello'"]


def test_read_file_can_inspect_tail_of_large_logs(tmp_path):
    root, tools = make_tools(tmp_path)
    (root / "server.log").write_text(
        "old line\n" + ("x" * 520_000) + "\nlatest failure",
        encoding="utf-8",
    )

    result = json.loads(
        tools.execute("read_file", {"path": "server.log", "tail": True, "line_count": 5}, lambda _: True)
    )

    assert "earlier content omitted" in result["tail"]
    assert "latest failure" in result["tail"]


def test_file_replacement_requires_approval_and_unique_match(tmp_path):
    root, tools = make_tools(tmp_path)
    target = root / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    calls = []

    rejected = json.loads(
        tools.execute(
            "replace_in_file",
            {"path": "app.py", "old_text": "1", "new_text": "2"},
            lambda message: calls.append(message) or False,
        )
    )
    assert rejected["status"] == "cancelled"
    assert target.read_text(encoding="utf-8") == "value = 1\n"

    applied = json.loads(
        tools.execute(
            "replace_in_file",
            {"path": "app.py", "old_text": "1", "new_text": "2"},
            lambda _message: True,
        )
    )
    assert applied["status"] == "updated"
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert calls and "Type 'yes'" in calls[0]


def test_create_file_requires_approval_and_never_overwrites(tmp_path):
    root, tools = make_tools(tmp_path)
    target = root / "new.py"

    cancelled = json.loads(
        tools.execute(
            "create_file",
            {"path": "new.py", "content": "print('new')\n"},
            lambda _: False,
        )
    )
    assert cancelled["status"] == "cancelled"
    assert not target.exists()

    created = json.loads(
        tools.execute(
            "create_file",
            {"path": "new.py", "content": "print('new')\n"},
            lambda _: True,
        )
    )
    overwrite = json.loads(
        tools.execute(
            "create_file",
            {"path": "new.py", "content": "print('overwrite')\n"},
            lambda _: True,
        )
    )
    assert created["status"] == "created"
    assert "error" in overwrite
    assert target.read_text(encoding="utf-8") == "print('new')\n"


def test_symlink_cannot_escape_project_root(tmp_path):
    root, tools = make_tools(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "outside-link").symlink_to(outside)

    result = json.loads(
        tools.execute("read_file", {"path": "outside-link"}, lambda _: True)
    )

    assert "error" in result
    assert "inside the project" in result["error"]


def test_test_runner_is_allowlisted_and_requires_approval(tmp_path, monkeypatch):
    _, tools = make_tools(tmp_path)
    executed = []

    def fake_run(argv, **kwargs):
        executed.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "2 passed", "")

    monkeypatch.setattr("llmcli.coding_tools.subprocess.run", fake_run)
    rejected_command = json.loads(
        tools.execute("run_tests", {"command": "python3 -c print(1)"}, lambda _: True)
    )
    cancelled = json.loads(
        tools.execute("run_tests", {"command": "python3 -m pytest -q"}, lambda _: False)
    )
    passed = json.loads(
        tools.execute("run_tests", {"command": "python3 -m pytest -q"}, lambda _: True)
    )

    assert "error" in rejected_command
    assert cancelled["status"] == "cancelled"
    assert passed["status"] == "passed"
    assert executed[0][1]["shell"] is False
    assert executed[0][1]["cwd"] == tools.root
