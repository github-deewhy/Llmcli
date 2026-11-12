"""Local Markdown memory for user-wide and project-specific context."""

import hashlib
import re
from pathlib import Path

from llmcli.security import MAX_INSTRUCTION_BYTES, read_capped_text


class MemoryStore:
    def __init__(
        self,
        base_prompt: str = "",
        root: Path | None = None,
        project_root: Path | None = None,
    ):
        memory_root = root or (Path.home() / ".llmcli")
        project_root = (project_root or Path.cwd()).resolve()
        project_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", project_root.name).strip("-")
        project_hash = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:12]

        self.base_prompt = base_prompt
        self.paths = {
            "global": memory_root / "MEMORY.md",
            "project": memory_root / "projects" / f"{project_name}-{project_hash}" / "MEMORY.md",
        }

    def read(self) -> dict[str, str]:
        """Read existing memory files without creating or changing them."""
        memories = {}
        for scope, path in self.paths.items():
            if not path.exists():
                memories[scope] = ""
            elif not path.is_file():
                raise OSError(f"Memory path is not a file: {path}")
            else:
                memories[scope] = read_capped_text(path)
        return memories

    def compose_system_prompt(self, memories: dict[str, str] | None = None) -> str:
        sections = [self.base_prompt] if self.base_prompt else []
        for scope, content in (memories if memories is not None else self.read()).items():
            if content.strip():
                sections.append(f"## Persistent {scope} memory\n\n{content.strip()}")
        return "\n\n".join(sections)

    def add(self, scope: str, content: str) -> Path:
        """Append Markdown content to a selected local memory file."""
        if scope not in self.paths:
            raise ValueError(f"Unknown memory scope: {scope}")
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")

        path = self.paths[scope]
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        if len(existing.encode("utf-8")) + len(content.encode("utf-8")) > MAX_INSTRUCTION_BYTES:
            raise ValueError(
                f"{scope} memory would exceed {MAX_INSTRUCTION_BYTES} bytes. "
                "Clear or shorten it first (it is sent with every prompt)."
            )
        separator = "\n" if existing and not existing.endswith("\n\n") else ""
        with path.open("a", encoding="utf-8") as memory_file:
            if separator:
                memory_file.write(separator)
            memory_file.write(f"{content}\n")
        return path

    def clear(self, scope: str) -> None:
        """Clear one existing Markdown memory file without removing its path."""
        if scope not in self.paths:
            raise ValueError(f"Unknown memory scope: {scope}")

        path = self.paths[scope]
        if not path.exists():
            return
        if not path.is_file():
            raise OSError(f"Memory path is not a file: {path}")
        path.write_text("", encoding="utf-8")
