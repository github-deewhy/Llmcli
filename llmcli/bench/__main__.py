"""Command line for the benchmark harness: ``python -m llmcli.bench list|check|run`` (spec 15, 2.1)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmcli.bench import CONFIGURATION, DEFAULT_RESULTS_DIR, DEFAULT_TASKS_DIR
from llmcli.bench.environment import machine_info, ollama_server_info, profile_info
from llmcli.bench.report import build_report, write_report
from llmcli.bench.runner import DEFAULT_DEADLINE_S, DEFAULT_MAX_TOKENS, Pacer, RunRecord, run_task
from llmcli.bench.tasks import Task, TaskError, load_oracle, load_tasks, null_script, task_hash
from llmcli.providers.fake import ScriptedProvider

FAKE_PROFILE = "fake"
FAKE_CONTEXT_SIZE = 8192
ACK_FLAG = "--i-understand-this-runs-model-code"
ACK_MESSAGE = (
    "Benchmark runs approve every edit and test automatically, so code written by the model runs on "
    f"this machine. Use a disposable machine or container, then pass {ACK_FLAG}."
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llmcli.bench", description="llmcli benchmark harness")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR, help="folder holding the tasks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the tasks")
    check = sub.add_parser("check", help="fake tier: every oracle passes and every null run fails")
    check.add_argument("--tasks", help="comma-separated task ids (default: all)")

    run = sub.add_parser("run", help="run tasks and write a report")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", action="append", help="profile from ~/.llmcli/config.toml (repeatable)")
    source.add_argument("--fake", action="store_true", help="replay each task's oracle script (no model)")
    run.add_argument("--tasks", help="comma-separated task ids (default: all)")
    run.add_argument("--repeats", type=_positive_int, default=3)
    run.add_argument("--deadline", type=_positive_float, default=DEFAULT_DEADLINE_S, help="seconds per task")
    run.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS, help="overrides the profile")
    run.add_argument(
        "--rpm", type=_positive_int, help="at most N model requests per minute (waiting is not timed)"
    )
    run.add_argument("--out", type=Path, help="report folder (default: bench-results/<time>-<commit>)")
    run.add_argument(ACK_FLAG, dest="acknowledged", action="store_true", help=argparse.SUPPRESS)
    return parser


def _task_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def cmd_list(tasks: list[Task]) -> int:
    for task in tasks:
        print(f"{task.id}  {task.category:<13} {task.title}")
    return 0


def check_tasks(tasks: list[Task]) -> list[tuple[Task, bool, bool]]:
    """(task, oracle passed, null failed) for each task."""
    results = []
    for task in tasks:
        oracle = run_task(task, ScriptedProvider(load_oracle(task)), FAKE_PROFILE, context_size=FAKE_CONTEXT_SIZE)
        null = run_task(task, ScriptedProvider(null_script()), FAKE_PROFILE, context_size=FAKE_CONTEXT_SIZE)
        results.append((task, oracle.passed, not null.passed))
    return results


def cmd_check(tasks: list[Task]) -> int:
    ok = True
    for task, oracle_passed, null_failed in check_tasks(tasks):
        good = oracle_passed and null_failed
        ok = ok and good
        print(
            f"{task.id}  oracle {'pass' if oracle_passed else 'FAIL'}  "
            f"null {'fails' if null_failed else 'PASSES'}  {'ok' if good else 'BROKEN'}"
        )
    print("All tasks ok." if ok else "Some tasks are broken.")
    return 0 if ok else 1


def _out_dir(args: argparse.Namespace, started: datetime, commit: str | None) -> Path:
    if args.out:
        return Path(args.out)
    return DEFAULT_RESULTS_DIR / f"{started:%Y%m%dT%H%M%SZ}-{(commit or 'nogit')[:7]}"


def _progress(record: RunRecord) -> None:
    status = "PASS" if record.passed else "fail"
    print(
        f"{record.task_id}  {record.profile}  #{record.repeat}  {status}  "
        f"{record.wall_s:.1f} s  {record.outcome.stop_reason}",
        flush=True,
    )


def _run_profile(
    name: str,
    args: argparse.Namespace,
    tasks: list[Task],
    records: list[RunRecord],
    profiles: list[dict[str, Any]],
) -> None:
    """Run every task for one profile. Raises ValueError for config or connection problems."""
    from llmcli.config import load_config
    from llmcli.providers import create_provider

    config = load_config({}, profile=name)
    config.max_tokens = args.max_tokens
    provider = create_provider(config)
    pacer = Pacer(args.rpm) if args.rpm else None
    try:
        healthy, message = provider.check_health()
        if not healthy:
            raise ValueError(f"Profile {name!r} is not reachable: {message}")
        provider.probe()
        entry = profile_info(name, config)
        entry.update(ollama_server_info(provider) if config.provider == "ollama" else {})
        profiles.append(entry)
        for task in tasks:
            for repeat in range(1, args.repeats + 1):
                record = run_task(
                    task,
                    provider,
                    name,
                    context_size=config.context_size,
                    tool_mode=config.tool_mode,
                    repeat=repeat,
                    deadline_s=args.deadline,
                    pacer=pacer,
                )
                records.append(record)
                _progress(record)
    finally:
        provider.close()


def _run_fake(args: argparse.Namespace, tasks: list[Task], records: list[RunRecord]) -> dict[str, Any]:
    for task in tasks:
        for repeat in range(1, args.repeats + 1):
            record = run_task(
                task,
                ScriptedProvider(load_oracle(task)),
                FAKE_PROFILE,
                context_size=FAKE_CONTEXT_SIZE,
                repeat=repeat,
                deadline_s=args.deadline,
            )
            records.append(record)
            _progress(record)
    return {"name": FAKE_PROFILE, "provider": "fake", "model": "fake-model", "endpoint": "fake://",
            "context_size": FAKE_CONTEXT_SIZE, "max_tokens": None, "temperature": None,
            "tool_mode": "auto", "extra_body": {}, "server_version": None, "model_digest": None}


def cmd_run(args: argparse.Namespace, tasks: list[Task]) -> int:
    if args.profile and not args.acknowledged:
        print(ACK_MESSAGE, file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    machine = machine_info()
    environment: dict[str, Any] = {
        "machine": machine,
        "profiles": [],
        "settings": {
            "repeats": args.repeats,
            "deadline_s": args.deadline,
            "max_tokens": None if args.fake else args.max_tokens,
            "rpm": None if args.fake else args.rpm,
        },
        "tasks": [{"id": t.id, "category": t.category, "hash": task_hash(t)} for t in tasks],
    }
    records: list[RunRecord] = []
    out_dir = _out_dir(args, started, machine.get("commit"))
    code = 0
    try:
        if args.fake:
            environment["profiles"].append(_run_fake(args, tasks, records))
        else:
            for name in args.profile:
                _run_profile(name, args, tasks, records, environment["profiles"])
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("Stopped by the user.", file=sys.stderr)
        code = 130
    if not records:
        return code
    report = build_report(environment, records, started.strftime("%Y-%m-%d %H:%M:%S"), CONFIGURATION)
    write_report(out_dir, report, records)
    passed = sum(1 for r in records if r.passed)
    print(f"{passed}/{len(records)} runs passed. Report: {out_dir / 'report.md'}")
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = load_tasks(args.tasks_dir, _task_ids(getattr(args, "tasks", None)))
    except TaskError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args.command == "list":
        return cmd_list(tasks)
    if args.command == "check":
        try:
            return cmd_check(tasks)
        except TaskError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
    try:
        return cmd_run(args, tasks)
    except TaskError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
