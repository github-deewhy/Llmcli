"""Regression tests for config precedence, provider quirks, and @file resolution."""

import pytest

from llmcli.config import Config, load_config
from llmcli.file_resolver import MAX_ATTACHMENT_BYTES, resolve_attachments
from llmcli.providers.ollama import OllamaProvider
from llmcli.providers.openai_compat import OpenAICompatProvider


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Empty HOME and cwd, with no LLMCLI_* variables set."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    for key in list(__import__("os").environ):
        if key.startswith("LLMCLI_"):
            monkeypatch.delenv(key)
    return home, project


def make_config(provider="ollama", model="qwen2.5-coder", endpoint="http://localhost:11434"):
    return Config(provider, model, endpoint, "", 8192, 1024, "CONTEXT.md")


# --- config precedence -----------------------------------------------------

def test_os_env_beats_cwd_dotenv(isolated, monkeypatch):
    _, project = isolated
    (project / ".env").write_text("LLMCLI_MODEL=from-dotenv\n")
    monkeypatch.setenv("LLMCLI_MODEL", "from-os-env")
    assert load_config().model == "from-os-env"


def test_cwd_dotenv_beats_home_dotenv(isolated):
    home, project = isolated
    (home / ".llmcli").mkdir()
    (home / ".llmcli" / ".env").write_text("LLMCLI_MODEL=home\nLLMCLI_THEME=forest\n")
    (project / ".env").write_text("LLMCLI_MODEL=project\n")
    config = load_config()
    assert config.model == "project"
    assert config.theme == "forest"  # falls through to the home file


def test_cli_override_beats_everything(isolated, monkeypatch):
    monkeypatch.setenv("LLMCLI_MODEL", "from-os-env")
    assert load_config({"model": "cli"}).model == "cli"


def test_load_config_does_not_mutate_environment(isolated):
    import os
    _, project = isolated
    (project / ".env").write_text("LLMCLI_MODEL=from-dotenv\n")
    load_config()
    assert "LLMCLI_MODEL" not in os.environ


# --- providers --------------------------------------------------------------

@pytest.mark.parametrize(
    "model, installed, expected",
    [
        ("qwen2.5-coder", ["qwen2.5-coder:latest"], True),
        ("qwen2.5-coder", ["qwen2.5-coder:7b"], False),
        ("qwen2.5-coder:7b", ["qwen2.5-coder:7b"], True),
        ("llama3", ["qwen2.5-coder:latest"], False),
    ],
)
def test_ollama_model_matching(model, installed, expected):
    provider = OllamaProvider(make_config(model=model))
    assert provider._model_installed(installed) is expected


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1"),
        ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/",
         "https://generativelanguage.googleapis.com/v1beta/openai"),
        ("http://localhost:1234", "http://localhost:1234/v1"),
    ],
)
def test_openai_compat_base_url(endpoint, expected):
    provider = OpenAICompatProvider(make_config("openai_compat", endpoint=endpoint))
    assert provider.base_url == expected


def test_extra_body_and_sampling_reach_the_request(monkeypatch):
    config = make_config("openai_compat", endpoint="https://integrate.api.nvidia.com/v1")
    config.temperature, config.top_p = 1.0, 0.95
    config.extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    sent = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "hi"}}]}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr("llmcli.providers.openai_compat.httpx.post", fake_post)
    OpenAICompatProvider(config).complete([{"role": "user", "content": "x"}], [])
    assert sent["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert sent["body"]["temperature"] == 1.0
    assert sent["body"]["top_p"] == 0.95
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_config_parses_sampling_and_extra_body(isolated, monkeypatch):
    monkeypatch.setenv("LLMCLI_TEMPERATURE", "1")
    monkeypatch.setenv("LLMCLI_TOP_P", "0.95")
    monkeypatch.setenv("LLMCLI_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": true}}')
    config = load_config()
    assert (config.temperature, config.top_p) == (1.0, 0.95)
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_config_rejects_bad_extra_body(isolated, monkeypatch):
    monkeypatch.setenv("LLMCLI_EXTRA_BODY", "{nope")
    with pytest.raises(ValueError, match="EXTRA_BODY"):
        load_config()


# --- @file attachments ---------------------------------------------------------

def test_email_addresses_are_untouched(isolated):
    text = "mail bob@example.com about it"
    assert resolve_attachments(text) == text


def test_missing_file_reference_is_untouched(isolated):
    text = "look at @nope/missing.py please"
    assert resolve_attachments(text) == text


def test_existing_file_is_attached(isolated):
    _, project = isolated
    (project / "a.py").write_text("print('hi')\n")
    result = resolve_attachments("explain @a.py")
    assert "print('hi')" in result and "```py" in result


def test_env_and_key_files_are_not_attached(isolated):
    _, project = isolated
    (project / ".env").write_text("LLMCLI_API_KEY=supersecret\n")
    (project / "server.pem").write_text("PRIVATE")
    for ref in ("@./.env", "@server.pem"):
        result = resolve_attachments(ref)
        assert "supersecret" not in result and "PRIVATE" not in result
        assert "Not attached" in result


def test_ignored_directories_are_not_attached(isolated):
    _, project = isolated
    (project / ".git").mkdir()
    (project / ".git" / "config.txt").write_text("gitsecret")
    assert "gitsecret" not in resolve_attachments("@.git/config.txt")


def test_oversized_files_are_not_attached(isolated):
    _, project = isolated
    (project / "big.txt").write_text("x" * (MAX_ATTACHMENT_BYTES + 1))
    result = resolve_attachments("@big.txt")
    assert "Not attached" in result and "xxxx" not in result
