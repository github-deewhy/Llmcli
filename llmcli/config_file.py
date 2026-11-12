"""Read and validate the single configuration file, ``~/.llmcli/config.toml``.

This module only parses and checks the file. Precedence and the final
``Config`` object live in ``llmcli.config``. Nothing here reads ``.env`` files
or environment-variable values (D-018).
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
PROVIDERS = ("ollama", "openai_compat")
TOOL_MODES = ("auto", "native", "text")

# Where each setting may appear (spec 13, section 2.2).
DEFAULTS_ONLY = frozenset({"profile", "theme", "context_file"})
PROFILE_ONLY = frozenset({"provider", "endpoint", "model", "api_key"})
SHARED = frozenset(
    {"context_size", "max_tokens", "temperature", "top_p", "extra_body", "request_timeout", "tool_mode"}
)
DEFAULTS_KEYS = DEFAULTS_ONLY | SHARED
PROFILE_KEYS = PROFILE_ONLY | SHARED
REQUIRED_IN_PROFILE = ("provider", "endpoint", "model")

STRING_KEYS = frozenset({"profile", "theme", "context_file", "provider", "endpoint", "model", "tool_mode"})
INT_KEYS = frozenset({"context_size", "max_tokens", "request_timeout"})
FLOAT_KEYS = frozenset({"temperature", "top_p"})
TABLE_KEYS = frozenset({"extra_body"})

LEGACY_FILES = ("profiles.toml", ".env")


def config_dir() -> Path:
    return Path.home() / ".llmcli"


def config_path() -> Path:
    return config_dir() / "config.toml"


def display_path(path: Path) -> str:
    """Show a path under the home directory as ``~/...``."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ConfigFile:
    """The parsed configuration file. ``exists`` is False when there is no file."""

    path: Path
    exists: bool = False
    defaults: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return display_path(self.path)


def _check_value(where: str, key: str, value: Any) -> None:
    """Raise ValueError if ``value`` has the wrong type for ``key``. Never echoes ``api_key``."""
    if key == "api_key":
        ok, expected = isinstance(value, str), "a string"
    elif key in STRING_KEYS:
        ok, expected = isinstance(value, str) and value.strip() != "", "a non-empty string"
    elif key in INT_KEYS:
        ok = isinstance(value, int) and not isinstance(value, bool) and value > 0
        expected = "a whole number greater than zero"
    elif key in FLOAT_KEYS:
        ok, expected = isinstance(value, (int, float)) and not isinstance(value, bool), "a number"
    else:  # TABLE_KEYS
        ok, expected = isinstance(value, dict), "a table, e.g. { key = value }"
    if not ok:
        raise ValueError(f"{where}: {key} must be {expected}.")
    if key == "provider" and value not in PROVIDERS:
        raise ValueError(f"{where}: provider must be one of {', '.join(PROVIDERS)}, got {value!r}.")
    if key == "tool_mode" and value not in TOOL_MODES:
        raise ValueError(f"{where}: tool_mode must be one of {', '.join(TOOL_MODES)}, got {value!r}.")
    if key == "endpoint":
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"{where}: endpoint must be an http:// or https:// URL, got {value!r}.")


def _check_section(where: str, values: Any, allowed: frozenset[str], wrong_place: str) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{where} must be a table of settings.")
    for key in sorted(values):
        if key not in allowed:
            if key in DEFAULTS_KEYS | PROFILE_KEYS:
                raise ValueError(f"{where}: {key} is not allowed here; {wrong_place}")
            raise ValueError(
                f"{where}: unknown setting {key!r}. Allowed: {', '.join(sorted(allowed))}."
            )
        _check_value(where, key, values[key])
    return dict(values)


def _check_profile(label: str, name: str, values: Any) -> dict[str, Any]:
    where = f"{label} [profiles.{name}]"
    if not PROFILE_NAME.fullmatch(name):
        raise ValueError(f"{label}: invalid profile name {name!r}; use letters, digits, '.', '_' or '-'.")
    profile = _check_section(where, values, PROFILE_KEYS, "put it in [defaults].")
    missing = [key for key in REQUIRED_IN_PROFILE if key not in profile]
    if missing:
        raise ValueError(f"{where}: missing required setting(s): {', '.join(missing)}.")
    return profile


def parse_config_text(text: str, path: Path) -> ConfigFile:
    """Parse and validate the file contents. Raises ValueError with a one-line message."""
    label = display_path(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        # The decoder message names line and column only, never the content.
        raise ValueError(f"{label}: not valid TOML: {error}") from None

    for key in sorted(data):
        if key == "default":
            raise ValueError(
                f"{label}: 'default' at the top level is not used; "
                'write profile = "<name>" inside [defaults].'
            )
        if key not in ("defaults", "profiles"):
            raise ValueError(
                f"{label}: unknown top-level entry {key!r}; only [defaults] and "
                "[profiles.<name>] tables are allowed."
            )

    defaults = _check_section(
        f"{label} [defaults]",
        data.get("defaults", {}),
        DEFAULTS_KEYS,
        "put it in a [profiles.<name>] table.",
    )
    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError(f"{label}: 'profiles' must be [profiles.<name>] tables.")
    profiles = {name: _check_profile(label, name, values) for name, values in raw_profiles.items()}

    chosen = defaults.get("profile")
    if chosen is not None and chosen not in profiles:
        available = ", ".join(sorted(profiles)) or "none"
        raise ValueError(
            f"{label} [defaults]: profile {chosen!r} is not defined. Available: {available}."
        )
    return ConfigFile(path=path, exists=True, defaults=defaults, profiles=profiles)


def load_config_file(path: Path | None = None) -> ConfigFile:
    """Read the configuration file; a missing file gives an empty ``ConfigFile``."""
    path = path or config_path()
    if not path.is_file():
        return ConfigFile(path=path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Could not read {display_path(path)}: {error.__class__.__name__}.") from None
    return parse_config_text(text, path)


def summarize_profiles(file: ConfigFile) -> list[dict[str, Any]]:
    """Rows for display. The key itself is never included, only whether one is set."""
    return [
        {
            "name": name,
            "provider": values["provider"],
            "model": values["model"],
            "endpoint": values["endpoint"],
            "key_set": bool(values.get("api_key")),
        }
        for name, values in sorted(file.profiles.items())
    ]


def key_permission_warning(file: ConfigFile) -> str | None:
    """Warn when the file holds an API key and other users can read it (spec 2.6)."""
    if not file.exists or not any(p.get("api_key") for p in file.profiles.values()):
        return None
    try:
        mode = file.path.stat().st_mode
    except OSError:
        return None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return (
            f"Warning: {file.label} contains an API key and is readable by other users; "
            f"run: chmod 600 {file.label}"
        )
    return None


def legacy_notice(environ_names: Iterable[str] | None = None, home_dir: Path | None = None) -> str | None:
    """One line naming old configuration that is now ignored (spec 2.7).

    Only the *names* of environment variables are inspected, never their values.
    A project's own ``./.env`` is deliberately not checked.
    """
    home_dir = home_dir or config_dir()
    names = os.environ.keys() if environ_names is None else environ_names
    found = [display_path(home_dir / name) for name in LEGACY_FILES if (home_dir / name).exists()]
    env_names = sorted(name for name in names if name.startswith("LLMCLI_"))
    if env_names:
        found.append("environment variables " + ", ".join(env_names))
    if not found:
        return None
    return (
        f"Note: ignoring old configuration ({'; '.join(found)}). Settings are now read only from "
        f"{display_path(config_path())}; see config.toml.example."
    )


def startup_messages(file: ConfigFile) -> list[str]:
    """Warnings to print once at start-up."""
    return [message for message in (legacy_notice(), key_permission_warning(file)) if message]
