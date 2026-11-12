"""Tests for configuration loading precedence."""

import os
import tempfile
from pathlib import Path

from llmcli.config import Config, load_config


def test_config_defaults():
    """Test that hardcoded defaults are applied when no env vars exist."""
    # Ensure no interfering env vars
    for key in ["LLMCLI_PROVIDER", "LLMCLI_MODEL", "LLMCLI_ENDPOINT"]:
        if key in os.environ:
            del os.environ[key]

    config = load_config()
    assert config.provider == "ollama"
    assert config.model == "qwen2.5-coder"
    assert config.endpoint == "http://localhost:11434"
    assert config.context_size == 8192


def test_config_env_override(monkeypatch, tmp_path):
    """Test that OS environment variables override defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLMCLI_PROVIDER", "openai_compat")
    monkeypatch.setenv("LLMCLI_MODEL", "gpt-4o")

    config = load_config()
    assert config.provider == "openai_compat"
    assert config.model == "gpt-4o"


def test_config_cli_override():
    """Test that CLI overrides have the highest precedence."""
    os.environ["LLMCLI_MODEL"] = "llama3"

    config = load_config(cli_overrides={"model": "qwen2.5-coder"})
    assert config.model == "qwen2.5-coder"

    # Cleanup
    del os.environ["LLMCLI_MODEL"]


def test_config_cli_override_for_theme():
    config = load_config(cli_overrides={"theme": "sunset"})
    assert config.theme == "sunset"


def test_config_dotenv_loading():
    """Test that .env files are loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / ".env"
        env_file.write_text("LLMCLI_PROVIDER=openai_compat\nLLMCLI_MODEL=test-model")

        original_cwd = Path.cwd()
        os.chdir(tmpdir)
        try:
            if "LLMCLI_PROVIDER" in os.environ:
                del os.environ["LLMCLI_PROVIDER"]

            config = load_config()
            assert config.provider == "openai_compat"
            assert config.model == "test-model"
        finally:
            os.chdir(original_cwd)
