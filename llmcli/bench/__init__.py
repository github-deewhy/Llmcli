"""Benchmark harness (spec 15, M2 part 1). A development tool, run with ``python -m llmcli.bench``.

It is deliberately not a ``llmcli`` subcommand (decision B1): benchmark runs
approve every edit and test automatically, which the product never does.
"""

from pathlib import Path

# Tasks live in the repository, not in the installed package (spec 15, 2.1).
DEFAULT_TASKS_DIR = Path(__file__).resolve().parents[2] / "bench" / "tasks"
DEFAULT_RESULTS_DIR = Path("bench-results")
CONFIGURATION = "m3b-v0.7"
