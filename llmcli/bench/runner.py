"""Run one task once: fresh workspace, unchanged agent, hidden checker (spec 15, sections 2.2 and 2.5)."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, cast

from rich.console import Console

from llmcli.agent import CodingAgent, RunOutcome
from llmcli.bench.tasks import Task
from llmcli.coding_tools import ProjectCodingTools
from llmcli.history import Conversation
from llmcli.providers.base import BaseProvider
from llmcli.providers.events import Event, ProviderError
from llmcli.ui import StreamingUI

SYSTEM_PROMPT = "You are a coding assistant. Use the tools to inspect and change files."
ANSWER_FILE = ".bench_answer.txt"
MAX_CHECK_OUTPUT = 20_000
DEFAULT_DEADLINE_S = 600.0
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class AgentSettings:
    """The agent settings a run uses (spec 16, part B); defaults are the 0.6 behaviour."""

    edit_matching: str = "exact"
    edit_tools: str = "replace"
    stop_guard: bool = False


class Pacer:
    """Keeps model requests at least ``60 / rpm`` seconds apart (spec 16, A5).

    One pacer is shared by every run of a profile, because a rate limit counts all
    requests to the server. ``clock`` and ``sleep`` can be replaced in tests.
    """

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rpm <= 0:
            raise ValueError("rpm must be a positive integer")
        self.interval_s = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None

    def wait(self) -> float:
        """Wait until the next request may start; return the seconds waited."""
        now = self._clock()
        waited = 0.0
        if self._last_start is not None:
            waited = max(0.0, self._last_start + self.interval_s - now)
            if waited > 0:
                self._sleep(waited)
        self._last_start = now + waited
        return waited


class PacedProvider:
    """Wraps a provider; waits for the pacer before each request and adds up the waiting time."""

    def __init__(self, inner: BaseProvider, pacer: Pacer):
        self._inner = inner
        self._pacer = pacer
        self.waited_s = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Iterator[Event]:
        self.waited_s += self._pacer.wait()
        yield from self._inner.stream(messages, tools)


class BenchDeadline(ProviderError):
    """The task ran past its deadline. Raised between stream events, so the agent stops cleanly."""


class DeadlineProvider:
    """Wraps a provider; its stream raises ``BenchDeadline`` once ``deadline`` (monotonic) has passed.

    The deadline is checked between events. A server that sends nothing at all
    is still bounded by the profile's ``request_timeout``.
    """

    def __init__(
        self,
        inner: BaseProvider,
        deadline: float,
        limit_s: float,
        paused_s: Callable[[], float] = lambda: 0.0,
    ):
        self._inner = inner
        self._deadline = deadline
        self._limit_s = limit_s
        self._paused_s = paused_s  # pacing time, which does not count against the deadline

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Iterator[Event]:
        self._check()
        events = self._inner.stream(messages, tools)
        try:
            for event in events:
                self._check()
                yield event
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    def _check(self) -> None:
        if time.monotonic() - self._paused_s() > self._deadline:
            raise BenchDeadline(f"Task deadline of {self._limit_s:g} s reached.")


@dataclass
class CheckResult:
    passed: bool
    exit_code: int | None
    output: str
    timed_out: bool = False


@dataclass
class RunRecord:
    """One task × profile × repeat."""

    task_id: str
    profile: str
    repeat: int
    passed: bool
    wall_s: float
    outcome: RunOutcome
    check: CheckResult
    trim_count: int
    context_size: int
    messages: list[dict[str, Any]] = field(default_factory=list)


def checker_environment(home: Path) -> dict[str, str]:
    """The only variables the checker sees (spec 15, 2.2)."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def checker_argv(task: Task) -> list[str]:
    """``python`` in a checker means the interpreter running the harness, so tests use the same pytest."""
    argv = list(task.checker)
    if argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    return argv


def run_checker(task: Task, work: Path, answer: str, home: Path) -> CheckResult:
    """Write the answer, copy the hidden checks over the workspace, run the checker. Pass = exit 0."""
    (work / ANSWER_FILE).write_text(answer, encoding="utf-8")
    shutil.copytree(task.check_dir, work, dirs_exist_ok=True)
    try:
        completed = subprocess.run(
            checker_argv(task),
            cwd=work,
            env=checker_environment(home),
            shell=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=task.checker_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(False, None, f"Checker timed out after {task.checker_timeout} s.", timed_out=True)
    except OSError as error:
        return CheckResult(False, None, f"Checker could not start: {error}")
    output = (completed.stdout + completed.stderr)[-MAX_CHECK_OUTPUT:]
    return CheckResult(completed.returncode == 0, completed.returncode, output)


def run_task(
    task: Task,
    provider: BaseProvider,
    profile: str,
    *,
    context_size: int,
    tool_mode: str = "auto",
    repeat: int = 1,
    deadline_s: float = DEFAULT_DEADLINE_S,
    pacer: Pacer | None = None,
    settings: AgentSettings | None = None,
) -> RunRecord:
    """Run ``task`` once with ``provider``. Every edit and test is approved automatically,
    which is acceptable only because the workspace is a throwaway copy (spec 15, 2.8).

    With a ``pacer``, time spent waiting for it counts neither in ``wall_s`` nor
    against the deadline."""
    with tempfile.TemporaryDirectory(prefix=f"llmcli-bench-{task.id}-") as tmp:
        tmp_path = Path(tmp)
        work = tmp_path / "work"
        shutil.copytree(task.repo_dir, work)
        home = tmp_path / "home"
        home.mkdir()

        ui = StreamingUI(Console(file=StringIO(), force_terminal=False, width=120))
        conversation = Conversation(SYSTEM_PROMPT, max_tokens=context_size)
        paced = PacedProvider(provider, pacer) if pacer is not None else None
        inner = cast(BaseProvider, paced) if paced is not None else provider

        def paused_s() -> float:
            return paced.waited_s if paced is not None else 0.0

        start = time.monotonic()
        wrapped = cast(BaseProvider, DeadlineProvider(inner, start + deadline_s, deadline_s, paused_s))
        settings = settings or AgentSettings()
        tools = ProjectCodingTools(work, ui, settings.edit_matching, settings.edit_tools)
        agent = CodingAgent(
            wrapped, conversation, tools, ui, lambda _message: True, tool_mode, settings.stop_guard
        )
        outcome = agent.run(task.prompt)
        wall_s = time.monotonic() - start - paused_s()

        check = run_checker(task, work, outcome.final_text, home)
        return RunRecord(
            task_id=task.id,
            profile=profile,
            repeat=repeat,
            passed=check.passed,
            wall_s=round(wall_s, 3),
            outcome=outcome,
            check=check,
            trim_count=conversation.trim_count,
            context_size=context_size,
            messages=copy.deepcopy(conversation.messages),
        )
