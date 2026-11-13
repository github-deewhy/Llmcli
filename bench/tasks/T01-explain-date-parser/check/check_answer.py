"""Pass if the final answer names the date-parsing function."""
import sys
from pathlib import Path

answer = Path(".bench_answer.txt").read_text(encoding="utf-8")
sys.exit(0 if "read_stamp" in answer else 1)
