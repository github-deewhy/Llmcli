"""Per-run metrics, per-cell aggregates, and the JSON/Markdown report (spec 15, section 2.7)."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from llmcli.bench.runner import RunRecord

# 2 (llmcli 0.6.0): adds tool_errors, per-run errors, per-cell common errors and the rpm setting.
# 3 (llmcli 0.7.0): adds repeat_skips, the agent settings per profile and settings.overrides.
SCHEMA_VERSION = 3


def _sum(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _mean(values: list[float | int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(mean(known), 3) if known else None


def run_metrics(record: RunRecord) -> dict[str, Any]:
    """The `08` metrics for one run that part 1 can measure."""
    usages = record.outcome.usages
    prompt = [u.prompt_tokens for u in usages]
    cached = [u.cached_tokens for u in usages]
    prompt_total = _sum(prompt)
    cached_total = _sum(cached)
    known_prompt = [p for p in prompt if p is not None]
    peak_prompt = max(known_prompt) if known_prompt else None
    outcome = record.outcome
    return {
        "passed": record.passed,
        "wall_s": record.wall_s,
        "stop_reason": outcome.stop_reason,
        "error_type": outcome.error_type,
        "rounds": outcome.rounds,
        "loop_guard_triggered": outcome.stop_reason == "round_limit",
        "tool_calls": outcome.tool_calls,
        "invalid_calls": outcome.invalid_calls,
        "tool_errors": outcome.tool_errors,
        "repeat_skips": outcome.repeat_skips,
        "tool_call_validity": (
            round((outcome.tool_calls - min(outcome.invalid_calls, outcome.tool_calls)) / outcome.tool_calls, 3)
            if outcome.tool_calls
            else None
        ),
        "repairs": outcome.repairs,
        "mode_switched": outcome.mode_switched,
        "prompt_tokens": prompt_total,
        "completion_tokens": _sum([u.completion_tokens for u in usages]),
        "cached_tokens": cached_total,
        "cache_hit_rate": (
            round(cached_total / prompt_total, 3) if cached_total is not None and prompt_total else None
        ),
        "ttft_s_first": usages[0].ttft_s if usages else None,
        "ttft_s_mean": _mean([u.ttft_s for u in usages]),
        "gen_tokens_per_s_mean": _mean([u.gen_tokens_per_s for u in usages]),
        "peak_prompt_tokens": peak_prompt,
        "peak_context_utilisation": (
            round(peak_prompt / record.context_size, 3) if peak_prompt is not None and record.context_size else None
        ),
        "trims": record.trim_count,
        "checker_exit_code": record.check.exit_code,
        "checker_timed_out": record.check.timed_out,
    }


def run_errors(record: RunRecord) -> dict[str, Any]:
    """Error messages of one run: the provider error, if the run ended with one, and each tool error."""
    outcome = record.outcome
    return {
        "provider_error": outcome.error if outcome.stop_reason == "provider_error" else None,
        "tool_errors": list(outcome.tool_error_messages),
    }


def _most_common(messages: list[str]) -> dict[str, Any] | None:
    """The most frequent message and how often it occurred; ties go to the first seen."""
    if not messages:
        return None
    message, count = Counter(messages).most_common(1)[0]
    return {"message": message, "count": count}


def _spread(values: list[Any]) -> dict[str, float | None]:
    known = [v for v in values if v is not None]
    if not known:
        return {"mean": None, "min": None, "max": None}
    return {"mean": round(mean(known), 3), "min": min(known), "max": max(known)}


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One cell per profile × task, in first-seen order."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        cells.setdefault((run["profile"], run["task_id"]), []).append(run)
    result = []
    for (profile, task_id), items in cells.items():
        metrics = [item["metrics"] for item in items]
        errors = [item.get("errors", {}) for item in items]
        result.append(
            {
                "profile": profile,
                "task_id": task_id,
                "runs": len(items),
                "passed": sum(1 for m in metrics if m["passed"]),
                "wall_s": _spread([m["wall_s"] for m in metrics]),
                "prompt_tokens": _spread([m["prompt_tokens"] for m in metrics]),
                "gen_tokens_per_s": _spread([m["gen_tokens_per_s_mean"] for m in metrics]),
                "rounds": _spread([m["rounds"] for m in metrics]),
                "tool_calls": sum(m["tool_calls"] for m in metrics),
                "invalid_calls": sum(m["invalid_calls"] for m in metrics),
                "tool_errors": sum(m["tool_errors"] for m in metrics),
                "repeat_skips": sum(m.get("repeat_skips", 0) for m in metrics),
                "loop_guard_triggers": sum(1 for m in metrics if m["loop_guard_triggered"]),
                "peak_context_utilisation": max(
                    (m["peak_context_utilisation"] for m in metrics if m["peak_context_utilisation"] is not None),
                    default=None,
                ),
                "trims": sum(m["trims"] for m in metrics),
                "stop_reasons": dict(Counter(m["stop_reason"] for m in metrics)),
                "common_provider_error": _most_common(
                    [e["provider_error"] for e in errors if e.get("provider_error")]
                ),
                "common_tool_error": _most_common([m for e in errors for m in e.get("tool_errors", [])]),
            }
        )
    return result


def transcript(record: RunRecord) -> dict[str, Any]:
    outcome = asdict(record.outcome)
    return {
        "task_id": record.task_id,
        "profile": record.profile,
        "repeat": record.repeat,
        "passed": record.passed,
        "outcome": outcome,
        "checker": asdict(record.check),
        "messages": record.messages,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_spread(spread: dict[str, Any]) -> str:
    if spread["mean"] is None:
        return "–"
    return f"{_fmt(spread['mean'])} ({_fmt(spread['min'])}–{_fmt(spread['max'])})"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _error_lines(cells: list[dict[str, Any]]) -> list[str]:
    """Per cell, the most common provider error and tool error, one line each (spec 16, A4)."""
    lines = []
    for cell in cells:
        for label, key in (("provider error", "common_provider_error"), ("tool error", "common_tool_error")):
            common = cell.get(key)
            if common:
                lines.append(f"- {cell['task_id']}, most common {label} ({common['count']}×): "
                             f"{_one_line(common['message'])}")
    if not lines:
        return []
    return ["Most common errors:", "", *lines, ""]


def markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    machine = env["machine"]
    lines = [
        f"# Benchmark report: {report['configuration']}",
        "",
        f"Started {report['started_utc']} UTC. Schema version {report['schema_version']}.",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| llmcli | {machine['llmcli_version']} |",
        f"| Commit | {_fmt(machine['commit'])}{' (uncommitted changes)' if machine['dirty'] else ''} |",
        f"| Python | {machine['python']} |",
        f"| OS | {machine['os']} |",
        f"| CPUs | {_fmt(machine['cpu_count'])} |",
        f"| RAM | {_fmt(round(machine['ram_bytes'] / 2**30, 1)) + ' GiB' if machine['ram_bytes'] else '–'} |",
        "| GPU | not recorded |",
        f"| Repeats | {env['settings']['repeats']} |",
        f"| Deadline per task | {env['settings']['deadline_s']:g} s |",
        f"| max_tokens override | {_fmt(env['settings']['max_tokens'])} |",
        f"| Requests per minute limit | {_fmt(env['settings'].get('rpm'))} |",
        f"| Agent setting overrides (--set) | "
        f"{', '.join(f'{k}={v}' for k, v in env['settings'].get('overrides', {}).items()) or '–'} |",
        "",
    ]
    for profile in env["profiles"]:
        lines += [
            f"## Profile `{profile['name']}`",
            "",
            f"Model `{profile.get('model')}` at `{profile.get('endpoint')}`, "
            f"provider {profile.get('provider')}, context_size {profile.get('context_size')}, "
            f"max_tokens {profile.get('max_tokens')}, temperature {_fmt(profile.get('temperature'))}, "
            f"tool_mode {profile.get('tool_mode')}, extra_body `{json.dumps(profile.get('extra_body', {}))}`, "
            f"edit_matching {profile.get('edit_matching')}, edit_tools {profile.get('edit_tools')}, "
            f"stop_guard {_fmt(profile.get('stop_guard'))}. "
            f"Server version {_fmt(profile.get('server_version'))}, model digest {_fmt(profile.get('model_digest'))}.",
            "",
            "| Task | Passed | Wall s mean (min–max) | Prompt tokens mean (min–max) | Gen tok/s mean (min–max) "
            "| Rounds mean (min–max) | Invalid / tool calls | Tool errors | Repeats skipped | Loop guard "
            "| Peak context | Trims | Stop reasons |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        cells = [cell for cell in report["cells"] if cell["profile"] == profile["name"]]
        for cell in cells:
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(cell["stop_reasons"].items()))
            lines.append(
                f"| {cell['task_id']} | {cell['passed']}/{cell['runs']} | {_fmt_spread(cell['wall_s'])} "
                f"| {_fmt_spread(cell['prompt_tokens'])} | {_fmt_spread(cell['gen_tokens_per_s'])} "
                f"| {_fmt_spread(cell['rounds'])} | {cell['invalid_calls']} / {cell['tool_calls']} "
                f"| {cell['tool_errors']} | {cell['repeat_skips']} | {cell['loop_guard_triggers']} | {_fmt(cell['peak_context_utilisation'])} "
                f"| {cell['trims']} | {reasons} |"
            )
        lines.append("")
        lines += _error_lines(cells)
    lines += [
        "## Tasks",
        "",
        "| Task | Category | Hash |",
        "| --- | --- | --- |",
        *[f"| {t['id']} | {t['category']} | `{t['hash']}` |" for t in env["tasks"]],
        "",
        "Transcripts are in `transcripts/`. Metrics the harness cannot measure yet are shown as –.",
        "",
    ]
    return "\n".join(lines)


def build_report(
    environment: dict[str, Any], records: list[RunRecord], started_utc: str, configuration: str
) -> dict[str, Any]:
    runs = [
        {"task_id": r.task_id, "profile": r.profile, "repeat": r.repeat, "metrics": run_metrics(r),
         "errors": run_errors(r)}
        for r in records
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": configuration,
        "started_utc": started_utc,
        "environment": environment,
        "runs": runs,
        "cells": aggregate(runs),
    }


def write_report(out_dir: Path, report: dict[str, Any], records: list[RunRecord]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    transcripts = out_dir / "transcripts"
    transcripts.mkdir(exist_ok=True)
    for record in records:
        profile = re.sub(r"[^A-Za-z0-9._-]", "_", record.profile)
        name = f"{record.task_id}-{profile}-{record.repeat}.json"
        (transcripts / name).write_text(
            json.dumps(transcript(record), indent=2, default=str) + "\n", encoding="utf-8"
        )
