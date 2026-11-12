"""Build the effective configuration.

Precedence (D-018, spec 13 section 2.3), highest first:
1. CLI flags (``--model``, ``--theme``; ``--profile`` picks the profile)
2. The active profile in ``~/.llmcli/config.toml``
3. The ``[defaults]`` table of that file
4. Built-in defaults

No ``.env`` file and no environment variable is read for settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmcli.config_file import ConfigFile, load_config_file

BUILT_IN_SOURCE = "built-in defaults"

BUILT_IN: dict[str, Any] = {
    "provider": "ollama",
    "endpoint": "http://localhost:11434",
    "model": "qwen2.5-coder",
    "api_key": "",
    "theme": "ocean",
    "context_file": "AGENTS.md",
    "context_size": 8192,
    "max_tokens": 4096,
    "temperature": None,
    "top_p": None,
    "extra_body": {},
}

CLI_KEYS = ("model", "theme")


@dataclass
class Config:
    provider: str
    model: str
    endpoint: str
    api_key: str
    context_size: int
    max_tokens: int
    context_file: str
    theme: str = "ocean"
    profile: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Extra top-level fields merged into OpenAI-compatible request bodies
    # (e.g. {"chat_template_kwargs": {"enable_thinking": true}}).
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Where the settings came from: the config file path or "built-in defaults".
    source: str = BUILT_IN_SOURCE


def select_profile(file: ConfigFile, requested: str | None) -> str | None:
    """Pick the active profile (spec 2.4). Returns None for the built-in local default."""
    names = sorted(file.profiles)
    if requested:
        if requested in file.profiles:
            return requested
        if names:
            raise ValueError(f"Unknown profile '{requested}'. Available: {', '.join(names)}.")
        raise ValueError(f"Unknown profile '{requested}'. No profiles are defined in {file.label}.")
    if file.defaults.get("profile"):
        return str(file.defaults["profile"])
    if len(names) == 1:
        return names[0]
    if len(names) > 1:
        raise ValueError(
            f"Several profiles are defined in {file.label} ({', '.join(names)}). "
            'Pass --profile <name> or set profile = "<name>" in [defaults].'
        )
    return None


def load_config(
    cli_overrides: dict[str, Any] | None = None,
    profile: str | None = None,
    file: ConfigFile | None = None,
) -> Config:
    """Return the effective configuration. Raises ValueError with a one-line message."""
    file = file if file is not None else load_config_file()
    active = select_profile(file, profile)

    values: dict[str, Any] = dict(BUILT_IN)
    values.update(file.defaults)
    values.pop("profile", None)
    if active is not None:
        values.update(file.profiles[active])
    for key in CLI_KEYS:
        if (cli_overrides or {}).get(key) is not None:
            values[key] = cli_overrides[key]  # type: ignore[index]

    temperature = values["temperature"]
    top_p = values["top_p"]
    return Config(
        provider=str(values["provider"]).lower(),
        model=str(values["model"]),
        endpoint=str(values["endpoint"]).rstrip("/"),
        api_key=str(values["api_key"]),
        context_size=int(values["context_size"]),
        max_tokens=int(values["max_tokens"]),
        context_file=str(values["context_file"]),
        theme=str(values["theme"]).lower(),
        profile=active,
        temperature=None if temperature is None else float(temperature),
        top_p=None if top_p is None else float(top_p),
        extra_body=dict(values["extra_body"]),
        source=file.label if file.exists else BUILT_IN_SOURCE,
    )
