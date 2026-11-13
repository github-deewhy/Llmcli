"""Pass if test_stack.py passes on the real Stack and fails on every mutant."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path.cwd()
TESTS = HERE / "test_stack.py"


def run_against(stack_source: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(stack_source, Path(tmp) / "stack.py")
        shutil.copy(TESTS, Path(tmp) / "test_stack.py")
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_stack.py"],
            cwd=tmp, capture_output=True, timeout=60, check=False,
        ).returncode


if not TESTS.is_file():
    print("test_stack.py was not written")
    sys.exit(1)
if run_against(HERE / "stack.py") != 0:
    print("test_stack.py fails on the correct Stack")
    sys.exit(1)
missed = [m.name for m in sorted((HERE / "mutants").glob("*.py")) if run_against(m) == 0]
if missed:
    print("not caught:", ", ".join(missed))
    sys.exit(1)
print("all mutants caught")
