"""D-019: the repository carries no cloud-vendor examples (spec 13, test 11)."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Built from parts so this file does not match itself.
VENDOR = re.compile(r"\b(" + "|".join(["gem" + "ini", "nvi" + "dia", "N" + "IM", "open" + "router"]) + r")\b",
                    re.IGNORECASE)
EXEMPT = {"CHANGELOG.md"}  # release history may name what was removed


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in out.splitlines() if name not in EXEMPT]


def test_no_cloud_vendor_names_outside_changelog():
    hits = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            match = VENDOR.search(line)
            if match:
                hits.append(f"{path.relative_to(ROOT)}:{number}: {match.group(0)}")
    assert hits == []
