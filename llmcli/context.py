"""Handles loading and parsing of the project context file."""

from pathlib import Path

from llmcli.config import Config


def load_context(config: Config) -> str:
    """
    Load the context file.
    Precedence: Current Working Directory > ~/.llmcli/ > ~/.
    """
    cwd_path = Path.cwd() / config.context_file
    home_dir = Path.home()
    home_paths = [
        home_dir / ".llmcli" / config.context_file,
        home_dir / config.context_file,
    ]

    if cwd_path.exists() and cwd_path.is_file():
        return cwd_path.read_text(encoding="utf-8")
    for home_path in home_paths:
        if home_path.exists() and home_path.is_file():
            return home_path.read_text(encoding="utf-8")

    return ""  # No context file found, which is acceptable
