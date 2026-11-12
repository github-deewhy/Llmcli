"""Project-scoped tools available to the interactive coding agent."""

import difflib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from rich.panel import Panel
from rich.syntax import Syntax

from llmcli.security import IGNORED_DIRECTORIES, filename_block_reason
from llmcli.ui import StreamingUI

MAX_FILE_BYTES = 512_000
MAX_SEARCH_RESULTS = 80
MAX_TEST_OUTPUT = 8_000
MAX_TOOL_RESULT = 24_000
MAX_READ_TEXT = 10_000
MAX_READ_LINE = 2_000
TEST_COMMAND_PREFIXES = (
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("npm", "test"),
    ("npm", "run", "test"),
    ("pnpm", "test"),
    ("yarn", "test"),
    ("cargo", "test"),
    ("go", "test"),
    ("swift", "test"),
    ("dotnet", "test"),
    ("make", "test"),
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List project files under a directory. Generated and secret directories are excluded.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Project-relative directory; defaults to the project root."}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a project file or log, optionally selecting a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "line_count": {"type": "integer", "minimum": 1, "maximum": 500},
                    "tail": {"type": "boolean", "description": "Read the end of a large log or file."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search project text files for a literal string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "description": "Project-relative directory; defaults to the project root."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace one exact, unique text block in a project file. Shows a diff and asks permission first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new project file. Shows its contents and asks permission first; cannot overwrite files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a recognized project test command after showing it and asking permission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A test command such as 'python3 -m pytest -q' or 'npm test'.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


class ProjectCodingTools:
    def __init__(self, project_root: Path, ui: StreamingUI):
        self.root = project_root.resolve()
        self.ui = ui

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        confirm: Callable[[str], bool],
    ) -> str:
        handlers = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_files": self._search_files,
            "replace_in_file": self._replace_in_file,
            "create_file": self._create_file,
            "run_tests": self._run_tests,
        }
        handler = handlers.get(name)
        if handler is None:
            return self._json({"error": f"Unknown tool: {name}"})
        try:
            result = handler(arguments, confirm)
        except (OSError, ValueError, UnicodeError) as error:
            result = {"error": str(error)}
        encoded = self._json(result)
        if len(encoded) > MAX_TOOL_RESULT:
            encoded = self._json({
                "truncated": True,
                "output": encoded[:MAX_TOOL_RESULT // 8],
                "note": "Tool output was truncated to fit the agent context.",
            })
        return encoded

    def _resolve_path(self, value: str, must_exist: bool = True) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A non-empty project-relative path is required.")
        raw_path = Path(value)
        target = (raw_path if raw_path.is_absolute() else self.root / raw_path).resolve()
        try:
            relative = target.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Paths must stay inside the project directory.") from error
        if any(part.lower() in IGNORED_DIRECTORIES for part in relative.parts):
            raise ValueError("Access to generated or version-control directories is blocked.")
        reason = filename_block_reason(target)
        if reason:
            raise ValueError(reason)
        if not must_exist and target.exists():
            raise ValueError("The requested new file already exists; use replace_in_file instead.")
        if must_exist and not target.exists():
            raise ValueError(f"File not found: {relative.as_posix()}")
        return target

    def _walk_files(self, start: Path):
        for current, directories, filenames in os.walk(start, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name.lower() not in IGNORED_DIRECTORIES
                and not (Path(current) / name).is_symlink()
            ]
            for filename in filenames:
                path = Path(current) / filename
                try:
                    self._resolve_path(str(path))
                except ValueError:
                    continue
                yield path

    @staticmethod
    def _validate_arguments(arguments: dict[str, Any], required: set[str], optional: set[str]) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        missing = required - arguments.keys()
        extra = arguments.keys() - required - optional
        if missing:
            raise ValueError(f"Missing tool arguments: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"Unexpected tool arguments: {', '.join(sorted(extra))}")

    def _list_files(self, arguments: dict[str, Any], _confirm: Callable[[str], bool]) -> dict[str, Any]:
        self._validate_arguments(arguments, set(), {"path"})
        start = self._resolve_path(arguments.get("path", ".")) if arguments.get("path") else self.root
        if not start.is_dir():
            raise ValueError("The requested path is not a directory.")
        files = []
        truncated = False
        for path in self._walk_files(start):
            if len(files) == 300:
                truncated = True
                break
            files.append(path.relative_to(self.root).as_posix())
        return {"files": files, "truncated": truncated}

    def _read_file(self, arguments: dict[str, Any], _confirm: Callable[[str], bool]) -> dict[str, Any]:
        self._validate_arguments(arguments, {"path"}, {"start_line", "line_count", "tail"})
        path = self._resolve_path(arguments["path"])
        if not path.is_file():
            raise ValueError("The requested path is not a file.")
        file_size = path.stat().st_size
        is_tail = arguments.get("tail", False)
        if not isinstance(is_tail, bool):
            raise ValueError("tail must be a boolean.")
        if file_size > MAX_FILE_BYTES and not is_tail:
            raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte read limit.")
        start_line = arguments.get("start_line", 1)
        line_count = arguments.get("line_count", 300)
        if not isinstance(start_line, int) or start_line < 1:
            raise ValueError("start_line must be a positive integer.")
        if not isinstance(line_count, int) or not 1 <= line_count <= 500:
            raise ValueError("line_count must be between 1 and 500.")
        if file_size > MAX_FILE_BYTES:
            with path.open("rb") as source:
                source.seek(file_size - MAX_FILE_BYTES)
                text = source.read(MAX_FILE_BYTES).decode("utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[-line_count:]
            numbered, truncated = self._format_lines(selected)
            numbered = f"[earlier content omitted: last {MAX_FILE_BYTES} bytes]\n{numbered}"
            return {
                "path": path.relative_to(self.root).as_posix(),
                "tail": numbered,
                "total_bytes": file_size,
                "truncated": truncated,
            }
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1:start_line - 1 + line_count]
        numbered, truncated = self._format_lines(selected, start_line)
        return {
            "path": path.relative_to(self.root).as_posix(),
            "lines": numbered,
            "total_lines": len(lines),
            "truncated": truncated,
        }

    def _search_files(self, arguments: dict[str, Any], _confirm: Callable[[str], bool]) -> dict[str, Any]:
        self._validate_arguments(arguments, {"query"}, {"path"})
        query = arguments["query"]
        if not isinstance(query, str) or not query:
            raise ValueError("Search query must be a non-empty string.")
        start = self._resolve_path(arguments["path"]) if arguments.get("path") else self.root
        if not start.is_dir():
            raise ValueError("The search path is not a directory.")
        matches = []
        truncated = False
        for path in self._walk_files(start):
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if query.casefold() in line.casefold():
                    matches.append(f"{path.relative_to(self.root).as_posix()}:{number}: {line[:500]}")
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        truncated = True
                        break
            if truncated:
                break
        return {"matches": matches, "truncated": truncated}

    def _replace_in_file(self, arguments: dict[str, Any], confirm: Callable[[str], bool]) -> dict[str, Any]:
        self._validate_arguments(arguments, {"path", "old_text", "new_text"}, set())
        path = self._resolve_path(arguments["path"])
        if not path.is_file():
            raise ValueError("The requested path is not a file.")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte edit limit.")
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        if not isinstance(old_text, str) or not old_text:
            raise ValueError("old_text must be a non-empty string.")
        if not isinstance(new_text, str):
            raise ValueError("new_text must be a string.")
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences != 1:
            raise ValueError(f"Expected old_text exactly once, found it {occurrences} times.")
        updated = original.replace(old_text, new_text, 1)
        relative = path.relative_to(self.root).as_posix()
        diff = self._diff(relative, original, updated)
        self.ui.print_diff(relative, diff)
        if not confirm(f"Apply this change to {relative}? Type 'yes' to approve: "):
            return {"status": "cancelled", "path": relative}
        if path.read_text(encoding="utf-8") != original:
            raise ValueError("File changed while approval was pending; no edit was applied.")
        path.write_text(updated, encoding="utf-8")
        return {"status": "updated", "path": relative, "diff": diff}

    def _create_file(self, arguments: dict[str, Any], confirm: Callable[[str], bool]) -> dict[str, Any]:
        self._validate_arguments(arguments, {"path", "content"}, set())
        path = self._resolve_path(arguments["path"], must_exist=False)
        content = arguments["content"]
        if not isinstance(content, str):
            raise ValueError("content must be a string.")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"New files are limited to {MAX_FILE_BYTES} bytes.")
        relative = path.relative_to(self.root).as_posix()
        diff = self._diff(relative, "", content)
        self.ui.print_diff(relative, diff)
        if not confirm(f"Create {relative}? Type 'yes' to approve: "):
            return {"status": "cancelled", "path": relative}
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._resolve_path(relative, must_exist=False)
        with path.open("x", encoding="utf-8") as new_file:
            new_file.write(content)
        return {"status": "created", "path": relative}

    def _run_tests(self, arguments: dict[str, Any], confirm: Callable[[str], bool]) -> dict[str, Any]:
        self._validate_arguments(arguments, {"command"}, set())
        command = arguments["command"]
        if not isinstance(command, str) or len(command) > 500:
            raise ValueError("Test command must be a string under 500 characters.")
        try:
            argv = shlex.split(command)
        except ValueError as error:
            raise ValueError(f"Invalid test command: {error}") from error
        if not argv or not any(argv[:len(prefix)] == list(prefix) for prefix in TEST_COMMAND_PREFIXES):
            raise ValueError("Only recognized test commands can be run.")
        display_command = shlex.join(argv)
        self.ui.print_info(f"Requested test command: [bold]{display_command}[/bold]")
        if not confirm(f"Run this test command in {self.root}? Type 'yes' to approve: "):
            return {"status": "cancelled", "command": display_command}
        try:
            result = subprocess.run(
                argv,
                cwd=self.root,
                shell=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
            return {
                "status": "timeout",
                "command": display_command,
                "output": self._truncate(stdout + stderr),
            }
        except OSError as error:
            return {"status": "failed_to_start", "command": display_command, "error": str(error)}
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "command": display_command,
            "exit_code": result.returncode,
            "output": self._truncate(result.stdout + result.stderr),
        }

    @staticmethod
    def _diff(path: str, before: str, after: str) -> str:
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}" if before else "/dev/null",
                tofile=f"b/{path}",
            )
        )

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_TEST_OUTPUT:
            return text
        return text[:MAX_TEST_OUTPUT] + "\n… output truncated …"

    @staticmethod
    def _format_lines(lines: list[str], start_line: int | None = None) -> tuple[str, bool]:
        output = []
        size = 0
        truncated = False
        for index, line in enumerate(lines, start_line or 1):
            if len(line) > MAX_READ_LINE:
                line = line[:MAX_READ_LINE] + " … line truncated"
                truncated = True
            rendered = f"{index}: {line}" if start_line is not None else line
            if size + len(rendered) > MAX_READ_TEXT:
                truncated = True
                break
            output.append(rendered)
            size += len(rendered) + 1
        if truncated:
            output.append("… additional content omitted …")
        return "\n".join(output), truncated

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)


def tool_schemas() -> list[dict[str, Any]]:
    return TOOL_DEFINITIONS
