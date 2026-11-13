"""Environment block for the report (spec 15, section 2.7). Every lookup is best effort: failures give None."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from llmcli import __version__
from llmcli.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _commit_from_files(root: Path) -> str | None:
    """Read HEAD without the git binary (for containers that have none)."""
    git_dir = root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:]
        ref_file = git_dir / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def git_info(root: Path | None = None) -> dict[str, Any]:
    """Commit and whether the tree has uncommitted changes (None when unknown)."""
    root = root or REPO_ROOT
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": _commit_from_files(root), "dirty": None}
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def ram_bytes() -> int | None:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=False
            ).stdout
            return int(out.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return None


def machine_info() -> dict[str, Any]:
    return {
        "llmcli_version": __version__,
        **git_info(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "gpu": None,  # not recorded in part 1 (spec 15, 2.7)
    }


def profile_info(name: str, config: Config) -> dict[str, Any]:
    """Settings that shape results. ``api_key`` is never included."""
    return {
        "name": name,
        "provider": config.provider,
        "endpoint": config.endpoint,
        "model": config.model,
        "context_size": config.context_size,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "extra_body": config.extra_body,
        "tool_mode": config.tool_mode,
        "request_timeout": config.request_timeout,
    }


def ollama_server_info(provider: Any) -> dict[str, Any]:
    """Ollama version (``GET /api/version``) and model digest (``GET /api/tags``) [VERIFY field names]."""
    info: dict[str, Any] = {"server_version": None, "model_digest": None}
    endpoint = getattr(provider, "endpoint", "")
    client = getattr(provider, "client", None)
    if client is None:
        return info
    try:
        response = client.get(f"{endpoint}/api/version")
        if response.status_code == 200:
            info["server_version"] = response.json().get("version")
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        response = client.get(f"{endpoint}/api/tags")
        if response.status_code == 200:
            wanted = {provider.model, f"{provider.model}:latest"}
            for model in response.json().get("models", []):
                if model.get("name") in wanted or model.get("model") in wanted:
                    info["model_digest"] = model.get("digest")
                    break
    except Exception:  # noqa: BLE001 - best effort
        pass
    return info
