"""Configuration loader with strict precedence chain."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from llmcli.profiles import load_profiles, profiles_path


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


def _env_file_values() -> dict[str, str | None]:
    """Merged values from ~/.llmcli/.env and ./.env (the latter wins)."""
    values: dict[str, str | None] = {}
    for env_path in (Path.home() / ".llmcli" / ".env", Path.cwd() / ".env"):
        if env_path.is_file():
            values.update(dotenv_values(env_path))
    return values


def lookup_env(name: str) -> str | None:
    """Look up a variable in the environment, then in the .env files."""
    value = os.getenv(name)
    return value if value is not None else _env_file_values().get(name)


def load_config(cli_overrides: dict | None = None, profile: str | None = None) -> Config:
    """
    Load configuration with the following precedence (highest to lowest):
    1. CLI overrides (e.g. --model)
    2. The active profile from ~/.llmcli/profiles.toml, if one is selected
    3. OS environment variables
    4. ./.env (current working directory)
    5. ~/.llmcli/.env (user home directory)
    6. Hardcoded defaults

    A profile is active when named via ``profile``, ``LLMCLI_PROFILE``, or the
    ``default`` entry of profiles.toml (checked in that order). A profile's API
    key comes only from the environment variable named by ``api_key_env``; a
    profile never falls back to LLMCLI_API_KEY, so one service's key is not
    sent to another service's endpoint.
    """
    cli_overrides = cli_overrides or {}

    # Step 1: Read dotenv files WITHOUT mutating os.environ, so real environment
    # variables always win over file values. ./.env beats ~/.llmcli/.env.
    file_values = _env_file_values()

    def env_value(name: str) -> str | None:
        value = os.getenv(name)
        return value if value is not None else file_values.get(name)

    # Step 2: Resolve the active profile (if any)
    default_profile, profiles = load_profiles()
    requested = profile or env_value("LLMCLI_PROFILE") or default_profile
    active: str | None = None
    profile_values: dict[str, Any] = {}
    if requested:
        if requested not in profiles:
            if profiles:
                hint = f"Available: {', '.join(sorted(profiles))}."
            else:
                hint = f"No profiles are defined; create {profiles_path()}."
            raise ValueError(f"Unknown profile '{requested}'. {hint}")
        active = requested
        profile_values = dict(profiles[requested])
        key_env = profile_values.pop("api_key_env", None)
        profile_values["api_key"] = ""
        if key_env:
            key = env_value(key_env)
            if not key:
                raise ValueError(
                    f"Profile '{requested}' reads its API key from {key_env}, which is not set. "
                    "Export it or add it to ~/.llmcli/.env."
                )
            profile_values["api_key"] = key

    # Step 3: Helper to get value with precedence
    def get_val(key: str, default: Any) -> Any:
        # CLI override
        if cli_overrides.get(key) is not None:
            return cli_overrides[key]
        # Active profile
        if key in profile_values:
            return profile_values[key]
        env_key = f"LLMCLI_{key.upper()}"
        # OS Env
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val
        # .env files (an empty value such as `LLMCLI_API_KEY=` is a real value)
        file_val = file_values.get(env_key)
        if file_val is not None:
            return file_val
        # Default
        return default

    def optional_float(key: str) -> float | None:
        value = get_val(key, None)
        if value is None or str(value).strip() == "":
            return None
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"LLMCLI_{key.upper()} must be a number, got {value!r}.") from error

    def positive_int(key: str, default: int) -> int:
        value = get_val(key, default)
        try:
            number = int(str(value).strip())
        except ValueError as error:
            raise ValueError(
                f"LLMCLI_{key.upper()} must be a whole number, got {value!r}."
            ) from error
        if number <= 0:
            raise ValueError(f"LLMCLI_{key.upper()} must be greater than zero, got {number}.")
        return number

    raw_extra = get_val("extra_body", "")
    extra_body: dict[str, Any] = {}
    if isinstance(raw_extra, dict):
        extra_body = raw_extra
    elif str(raw_extra).strip():
        try:
            extra_body = json.loads(str(raw_extra))
        except json.JSONDecodeError as error:
            raise ValueError(f"LLMCLI_EXTRA_BODY must be valid JSON: {error}") from error
        if not isinstance(extra_body, dict):
            raise ValueError("LLMCLI_EXTRA_BODY must be a JSON object.")

    return Config(
        provider=str(get_val("provider", "ollama")).lower(),
        model=str(get_val("model", "qwen2.5-coder")),
        endpoint=str(get_val("endpoint", "http://localhost:11434")).rstrip("/"),
        api_key=str(get_val("api_key", "")),
        context_size=positive_int("context_size", 8192),
        max_tokens=positive_int("max_tokens", 4096),
        context_file=str(get_val("context_file", "AGENTS.md")),
        theme=str(get_val("theme", "ocean")).lower(),
        temperature=optional_float("temperature"),
        top_p=optional_float("top_p"),
        extra_body=extra_body,
        profile=active,
    )
