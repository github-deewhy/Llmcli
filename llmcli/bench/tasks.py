"""Load benchmark tasks and their oracle scripts (spec 15, sections 2.2 and 2.4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmcli.providers.base_types import ToolCall
from llmcli.providers.fake import text_turn, tool_turn

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

CATEGORIES = ("explain", "fix-bug", "add-function", "refactor", "write-tests", "multi-file")
REQUIRED_KEYS = ("id", "title", "category", "prompt", "checker")
DEFAULT_CHECKER_TIMEOUT = 60


class TaskError(ValueError):
    """A task folder is malformed. The message is one line."""


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    category: str
    prompt: str
    checker: tuple[str, ...]
    checker_timeout: int
    path: Path

    @property
    def repo_dir(self) -> Path:
        return self.path / "repo"

    @property
    def check_dir(self) -> Path:
        return self.path / "check"

    @property
    def oracle_path(self) -> Path:
        return self.path / "oracle.json"


def load_task(path: Path) -> Task:
    """Read ``task.toml`` in ``path`` and check the folder layout."""
    name = path.name
    toml_path = path / "task.toml"
    if not toml_path.is_file():
        raise TaskError(f"Task {name}: task.toml is missing.")
    try:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise TaskError(f"Task {name}: task.toml is not valid TOML ({error}).") from error

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise TaskError(f"Task {name}: task.toml is missing {', '.join(missing)}.")
    for key in ("id", "title", "category", "prompt"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise TaskError(f"Task {name}: {key} must be a non-empty string.")
    if data["category"] not in CATEGORIES:
        raise TaskError(
            f"Task {name}: unknown category {data['category']!r}. Use one of: {', '.join(CATEGORIES)}."
        )
    checker = data["checker"]
    if not isinstance(checker, list) or not checker or not all(isinstance(a, str) and a for a in checker):
        raise TaskError(f"Task {name}: checker must be a non-empty list of strings.")
    timeout = data.get("checker_timeout", DEFAULT_CHECKER_TIMEOUT)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise TaskError(f"Task {name}: checker_timeout must be a positive integer.")
    for folder in ("repo", "check"):
        if not (path / folder).is_dir():
            raise TaskError(f"Task {name}: the {folder}/ folder is missing.")

    return Task(
        id=data["id"],
        title=data["title"],
        category=data["category"],
        prompt=data["prompt"],
        checker=tuple(checker),
        checker_timeout=timeout,
        path=path,
    )


def load_tasks(root: Path, ids: list[str] | None = None) -> list[Task]:
    """All tasks under ``root`` sorted by id, or only ``ids`` (in the order given)."""
    if not root.is_dir():
        raise TaskError(
            f"Task folder not found: {root}. Run from a checkout of the repository or pass --tasks-dir."
        )
    tasks = [load_task(child) for child in sorted(root.iterdir()) if child.is_dir()]
    seen: dict[str, Task] = {}
    for task in tasks:
        if task.id in seen:
            raise TaskError(f"Task id {task.id} is used by {seen[task.id].path.name} and {task.path.name}.")
        seen[task.id] = task
    if ids is None:
        return sorted(tasks, key=lambda t: t.id)
    unknown = [task_id for task_id in ids if task_id not in seen]
    if unknown:
        raise TaskError(f"Unknown task id: {', '.join(unknown)}. Available: {', '.join(sorted(seen))}.")
    return [seen[task_id] for task_id in ids]


def task_hash(task: Task) -> str:
    """SHA-256 over every file in the task folder (relative path and bytes), first 16 hex digits."""
    digest = hashlib.sha256()
    for file in sorted(p for p in task.path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(file.relative_to(task.path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def load_oracle(task: Task) -> list[list[Any]]:
    """Scripted turns for ``ScriptedProvider`` that solve the task (spec 15, 2.4)."""
    name = task.path.name
    if not task.oracle_path.is_file():
        raise TaskError(f"Task {name}: oracle.json is missing.")
    try:
        data = json.loads(task.oracle_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TaskError(f"Task {name}: oracle.json is not valid JSON ({error}).") from error
    if not isinstance(data, list) or not data:
        raise TaskError(f"Task {name}: oracle.json must be a non-empty list of turns.")

    turns: list[list[Any]] = []
    call_number = 0
    for index, turn in enumerate(data, start=1):
        if not isinstance(turn, dict):
            raise TaskError(f"Task {name}: oracle turn {index} must be an object.")
        text = turn.get("text", "")
        if not isinstance(text, str):
            raise TaskError(f"Task {name}: oracle turn {index} text must be a string.")
        raw_calls = turn.get("tool_calls")
        if raw_calls is None:
            turns.append(text_turn(text))
            continue
        if not isinstance(raw_calls, list) or not raw_calls:
            raise TaskError(f"Task {name}: oracle turn {index} tool_calls must be a non-empty list.")
        calls = []
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                raise TaskError(f"Task {name}: oracle turn {index} has a tool call without a name.")
            arguments = raw.get("arguments", {})
            if not isinstance(arguments, dict):
                raise TaskError(f"Task {name}: oracle turn {index} arguments must be an object.")
            call_number += 1
            calls.append(ToolCall(id=f"call_{call_number}", name=raw["name"], arguments=arguments))
        turns.append(tool_turn(*calls, text=text))
    return turns


def null_script() -> list[list[Any]]:
    """A run that changes nothing; every checker must fail it (spec 15, 2.4)."""
    return [text_turn("Done.")]
