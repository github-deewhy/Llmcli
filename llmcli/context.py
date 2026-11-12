"""Handles loading and parsing of the project instruction file."""

from pathlib import Path

from llmcli.config import Config
from llmcli.security import read_capped_text

DEFAULT_CONTEXT_FILE = "AGENTS.md"
LEGACY_CONTEXT_FILE = "CONTEXT.md"


def candidate_names(config: Config) -> list[str]:
    """Filenames to look for, in order. CONTEXT.md is a legacy alias for AGENTS.md."""
    names = [config.context_file]
    if config.context_file == DEFAULT_CONTEXT_FILE:
        names.append(LEGACY_CONTEXT_FILE)
    return names


def load_context(config: Config) -> str:
    """
    Load the project instruction file (AGENTS.md, or legacy CONTEXT.md).
    Precedence: Current Working Directory > ~/.llmcli/ > ~/.
    Within a location, the configured name wins over the legacy alias.
    File size is capped (see security.MAX_INSTRUCTION_BYTES).
    """
    home_dir = Path.home()
    folders = [Path.cwd(), home_dir / ".llmcli", home_dir]
    for folder in folders:
        for name in candidate_names(config):
            path = folder / name
            if path.is_file():
                return read_capped_text(path)
    return ""  # No instruction file found, which is acceptable
