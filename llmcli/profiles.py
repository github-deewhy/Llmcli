"""Named provider profiles stored in ``~/.llmcli/profiles.toml``."""

import re
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
STRING_KEYS = {"provider", "model", "endpoint", "api_key_env"}
INT_KEYS = {"context_size", "max_tokens"}
FLOAT_KEYS = {"temperature", "top_p"}
OBJECT_KEYS = {"extra_body"}
PROFILE_KEYS = STRING_KEYS | INT_KEYS | FLOAT_KEYS | OBJECT_KEYS
TOP_LEVEL_KEYS = {"default", "profiles"}


def profiles_path() -> Path:
    return Path.home() / ".llmcli" / "profiles.toml"


def _validate_profile(name: str, values: Any) -> dict[str, Any]:
    if not PROFILE_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid profile name {name!r}: use letters, digits, '.', '_' or '-'."
        )
    if not isinstance(values, dict):
        raise ValueError(f"Profile '{name}' must be a table of settings.")
    unknown = set(values) - PROFILE_KEYS
    if unknown:
        raise ValueError(
            f"Profile '{name}' has unknown setting(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(PROFILE_KEYS))}."
        )
    for key, value in values.items():
        if key in STRING_KEYS:
            ok = isinstance(value, str) and value.strip() != ""
            expected = "a non-empty string"
        elif key in INT_KEYS:
            ok = isinstance(value, int) and not isinstance(value, bool) and value > 0
            expected = "a positive integer"
        elif key in FLOAT_KEYS:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            expected = "a number"
        else:
            ok = isinstance(value, dict)
            expected = "a table"
        if not ok:
            raise ValueError(f"Profile '{name}': {key} must be {expected}.")
    if "api_key" in values:  # defensive; unreachable because of the unknown-key check
        raise ValueError(f"Profile '{name}': store keys in an env var via api_key_env.")
    return dict(values)


def load_profiles(path: Path | None = None) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Return ``(default_profile_name, {name: settings})``; empty if no file exists."""
    path = path or profiles_path()
    if not path.is_file():
        return None, {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Could not read {path}: {error}") from error

    unknown_top = set(data) - TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(
            f"{path}: unknown top-level key(s): {', '.join(sorted(unknown_top))}. "
            "Settings belong inside a [profiles.<name>] table."
        )
    raw = data.get("profiles", {})
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'profiles' must be a table of [profiles.<name>] sections.")

    profiles = {name: _validate_profile(name, values) for name, values in raw.items()}
    default = data.get("default")
    if default is not None and default not in profiles:
        raise ValueError(
            f"{path}: default profile {default!r} is not defined. "
            f"Available: {', '.join(sorted(profiles)) or 'none'}."
        )
    return default, profiles


def summarize_profiles(profiles: dict[str, dict[str, Any]], lookup_env) -> list[dict[str, Any]]:
    """Describe each profile for display; ``lookup_env(name)`` reports whether a key is set."""
    rows = []
    for name in sorted(profiles):
        values = profiles[name]
        key_env = values.get("api_key_env")
        rows.append(
            {
                "name": name,
                "provider": values.get("provider", "(from .env)"),
                "model": values.get("model", "(from .env)"),
                "endpoint": values.get("endpoint", "(from .env)"),
                "key_env": key_env,
                "key_set": bool(key_env and lookup_env(key_env)),
            }
        )
    return rows
