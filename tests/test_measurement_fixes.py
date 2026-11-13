"""Measurement fixes for M3 part A (spec 16, section 2): usage request, tokens/s, tool errors,
error lines in the report, request pacing."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from llmcli.bench.__main__ import main
from llmcli.bench.report import aggregate, build_report, markdown, run_errors, run_metrics
from llmcli.bench.runner import Pacer, run_task
from llmcli.bench.tasks import load_task
from llmcli.providers import base as base_module
from llmcli.providers.base import MIN_SPEED_SECONDS, MIN_SPEED_TOKENS, StreamState, ToolCall
from llmcli.providers.events import Capabilities, Done, Event, ProviderHTTPError, TextDelta, Usage
from llmcli.providers.fake import ScriptedProvider, text_turn, tool_turn
from llmcli.providers.openai_compat import OpenAICompatProvider
from tests.test_bench import make_agent, make_task
from tests.test_providers import make_config, sse, with_transport

ANSWER = (
    {"choices": [{"delta": {"content": "Hello"}}]},
    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
)


def openai_provider(handler, **extra):
    return with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h/v1", **extra)), handler)


# --- A1: ask for token counts, fall back once on HTTP 400 ----------------------------------

def test_usage_option_is_sent_and_usage_is_read():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text=sse(*ANSWER))

    provider = openai_provider(handler)
    events = list(provider.stream([{"role": "user", "content": "hi"}]))
    assert bodies[0]["stream_options"] == {"include_usage": True}
    usage = next(e for e in events if isinstance(e, Usage))
    assert (usage.prompt_tokens, usage.completion_tokens) == (12, 3)
    done = events[-1]
    assert isinstance(done, Done) and done.finish_reason == "stop"


def test_http_400_retries_once_without_usage_option_and_remembers():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "stream_options" in body:
            return httpx.Response(400, text='{"error": "unknown field stream_options"}')
        return httpx.Response(200, text=sse(*ANSWER))

    provider = openai_provider(handler)
    first = list(provider.stream([{"role": "user", "content": "hi"}]))
    assert [("stream_options" in b) for b in bodies] == [True, False]
    assert [e.text for e in first if isinstance(e, TextDelta)] == ["Hello"]  # no duplicated text
    assert not provider.send_usage_option

    list(provider.stream([{"role": "user", "content": "again"}]))
    assert [("stream_options" in b) for b in bodies] == [True, False, False]


def test_http_400_without_usage_option_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400)

    provider = openai_provider(handler)
    provider.send_usage_option = False
    with pytest.raises(ProviderHTTPError):
        list(provider.stream([{"role": "user", "content": "hi"}]))
    assert len(calls) == 1


def test_second_http_400_is_reported_not_retried_again():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400)

    provider = openai_provider(handler)
    with pytest.raises(ProviderHTTPError) as caught:
        list(provider.stream([{"role": "user", "content": "hi"}]))
    assert len(calls) == 2
    assert caught.value.status_code == 400 and "\n" not in str(caught.value)


def test_other_http_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    provider = openai_provider(handler)
    with pytest.raises(ProviderHTTPError):
        list(provider.stream([{"role": "user", "content": "hi"}]))
    assert len(calls) == 1 and provider.send_usage_option


def test_stream_options_from_extra_body_wins_and_is_not_dropped():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400)

    provider = openai_provider(handler, extra_body={"stream_options": {"include_usage": False}})
    with pytest.raises(ProviderHTTPError):
        list(provider.stream([{"role": "user", "content": "hi"}]))
    assert len(bodies) == 1
    assert bodies[0]["stream_options"] == {"include_usage": False}


# --- A2: tokens/s only when measurable ---------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    "tokens, elapsed, expected",
    [
        (500, 0.001, None),  # one burst: the old code reported 500,000 tokens/s
        (MIN_SPEED_TOKENS - 1, 5.0, None),  # too few tokens
        (MIN_SPEED_TOKENS, MIN_SPEED_SECONDS, round(MIN_SPEED_TOKENS / MIN_SPEED_SECONDS, 2)),
        (100, 2.0, 50.0),
        (None, 2.0, None),
    ],
)
def test_client_side_speed_needs_enough_tokens_and_time(monkeypatch, tokens, elapsed, expected):
    clock = FakeClock()
    monkeypatch.setattr(base_module.time, "monotonic", clock)
    state = StreamState(Capabilities())
    list(state.content("x"))
    clock.now += elapsed
    assert state.wall_tokens_per_s(tokens) == expected


# --- A3: failed tool executions are counted -----------------------------------------------

def test_failed_edit_counts_as_tool_error_not_invalid_call(tmp_path):
    bad_edit = ToolCall("c1", "replace_in_file", {"path": "app.py", "old_text": "nope", "new_text": "x"})
    read = ToolCall("c2", "read_file", {"path": "app.py"})
    agent, _ = make_agent(tmp_path, [tool_turn(bad_edit, read), tool_turn(bad_edit), text_turn("gave up")])
    outcome = agent.run("edit")
    assert (outcome.tool_calls, outcome.invalid_calls, outcome.tool_errors) == (3, 0, 2)
    assert len(outcome.tool_error_messages) == 2
    assert outcome.tool_error_messages[0].startswith("replace_in_file: ")
    assert "\n" not in outcome.tool_error_messages[0]


def test_schema_failure_is_invalid_call_not_tool_error(tmp_path):
    agent, _ = make_agent(tmp_path, [tool_turn(ToolCall("c1", "read_file", {})), text_turn("ok")])
    outcome = agent.run("read")
    assert (outcome.invalid_calls, outcome.tool_errors) == (1, 0)


def test_tool_errors_reach_metrics_cells_and_report(tmp_path):
    task = load_task(make_task(tmp_path, oracle=[
        {"tool_calls": [{"name": "replace_in_file",
                         "arguments": {"path": "app.py", "old_text": "missing", "new_text": "x"}}]},
        {"text": "Could not edit."},
    ]))
    records = [
        run_task(task, ScriptedProvider(json_turns(task)), "p", context_size=8192, repeat=n) for n in (1, 2)
    ]
    assert run_metrics(records[0])["tool_errors"] == 1
    assert run_errors(records[0])["tool_errors"][0].startswith("replace_in_file: ")
    report = build_report(empty_environment(), records, "t", "c")
    [cell] = report["cells"]
    assert cell["tool_errors"] == 2
    assert cell["common_tool_error"]["count"] == 2
    md = markdown(report)
    assert "| Tool errors |" in md
    assert "- X01, most common tool error (2×): replace_in_file: " in md


def json_turns(task):
    from llmcli.bench.tasks import load_oracle
    return load_oracle(task)


def empty_environment():
    from llmcli.bench.environment import machine_info
    return {
        "machine": machine_info(),
        "profiles": [{"name": "p", "model": "m", "endpoint": "e", "provider": "fake"}],
        "settings": {"repeats": 2, "deadline_s": 600.0, "max_tokens": None, "rpm": 40},
        "tasks": [],
    }


# --- A4: most common errors per cell ------------------------------------------------------

def test_most_common_provider_error_is_shown_once_per_cell():
    def run(error, stop="provider_error"):
        metrics = {"passed": False, "wall_s": 1.0, "prompt_tokens": None, "gen_tokens_per_s_mean": None,
                   "rounds": 1, "tool_calls": 0, "invalid_calls": 0, "tool_errors": 0,
                   "loop_guard_triggered": False, "peak_context_utilisation": None, "trims": 0,
                   "stop_reason": stop}
        return {"profile": "p", "task_id": "T01", "metrics": metrics,
                "errors": {"provider_error": error, "tool_errors": []}}

    cells = aggregate([run("HTTP 429"), run("HTTP 429"), run("timeout"), run(None, "answered")])
    assert cells[0]["common_provider_error"] == {"message": "HTTP 429", "count": 2}
    assert cells[0]["common_tool_error"] is None
    report = {"configuration": "c", "started_utc": "t", "schema_version": 2, "cells": cells,
              "environment": empty_environment()}
    md = markdown(report)
    assert md.count("most common provider error") == 1
    assert "- T01, most common provider error (2×): HTTP 429" in md
    assert "Requests per minute limit | 40" in md


def test_cells_without_errors_add_no_error_lines(tmp_path):
    task = load_task(make_task(tmp_path))
    record = run_task(task, ScriptedProvider(json_turns(task)), "p", context_size=8192)
    md = markdown(build_report(empty_environment(), [record], "t", "c"))
    assert "Most common errors" not in md


# --- A5: request pacing --------------------------------------------------------------------

class SleepRecorder:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.clock.now += seconds


def test_pacer_spaces_requests_without_real_sleeping():
    clock = FakeClock()
    sleep = SleepRecorder(clock)
    pacer = Pacer(40, clock=clock, sleep=sleep)  # 1.5 s apart
    assert pacer.wait() == 0.0
    clock.now += 0.5
    assert pacer.wait() == pytest.approx(1.0)
    clock.now += 2.0  # slower than the limit: no wait
    assert pacer.wait() == 0.0
    assert sleep.sleeps == [1.0]


def test_pacer_rejects_non_positive_rpm():
    with pytest.raises(ValueError):
        Pacer(0)


class ClockedProvider(ScriptedProvider):
    """Each request takes one simulated second."""

    def __init__(self, turns, clock: FakeClock):
        super().__init__(turns)
        self.clock = clock

    def stream(self, messages, tools=None) -> Iterator[Event]:
        self.clock.now += 1.0
        yield from super().stream(messages, tools)


def test_pacing_wait_is_excluded_from_wall_time_and_deadline(tmp_path, monkeypatch):
    from llmcli.bench import runner as runner_module
    clock = FakeClock()
    monkeypatch.setattr(runner_module.time, "monotonic", clock)
    sleep = SleepRecorder(clock)
    pacer = Pacer(6, clock=clock, sleep=sleep)  # 10 s apart
    task = load_task(make_task(tmp_path))
    provider = ClockedProvider(json_turns(task), clock)
    # Two requests: 2 s of model time and 9 s of pacing. A 5 s deadline must not trip.
    record = run_task(task, provider, "p", context_size=8192, deadline_s=5.0, pacer=pacer)
    assert sleep.sleeps == [9.0]
    assert record.outcome.stop_reason == "answered" and record.passed
    assert record.wall_s == pytest.approx(2.0)


def test_rpm_flag_reaches_report_and_paces(tmp_path, monkeypatch):
    make_task(tmp_path)
    from tests.test_bench import fake_config
    task = load_task(tmp_path / "tasks" / "X01-sample")
    waits = []
    monkeypatch.setattr("llmcli.config.load_config", lambda overrides, profile: fake_config())
    monkeypatch.setattr("llmcli.providers.create_provider", lambda config: ScriptedProvider(json_turns(task)))
    monkeypatch.setattr(Pacer, "wait", lambda self: waits.append(self.interval_s) or 0.0)
    out = tmp_path / "out"
    code = main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--profile", "p",
                 "--i-understand-this-runs-model-code", "--repeats", "1", "--rpm", "40", "--out", str(out)])
    assert code == 0
    assert waits == [1.5, 1.5]
    report = json.loads((out / "report.json").read_text())
    assert report["environment"]["settings"]["rpm"] == 40


def test_rpm_must_be_positive(tmp_path, capsys):
    make_task(tmp_path)
    with pytest.raises(SystemExit):
        main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--fake", "--rpm", "0"])


def test_fake_run_records_empty_errors(tmp_path):
    out = tmp_path / "out"
    make_task(tmp_path)
    assert main(["--tasks-dir", str(tmp_path / "tasks"), "run", "--fake", "--repeats", "1",
                 "--out", str(out)]) == 0
    run = json.loads((out / "report.json").read_text())["runs"][0]
    assert run["errors"] == {"provider_error": None, "tool_errors": []}
    assert Path(out / "report.md").read_text().count("Tool errors") == 1
