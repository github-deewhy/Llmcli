"""Profile commands and live switching with the single config file (spec 13, test 10)."""

from io import StringIO

import pytest
from rich.console import Console

from llmcli.cli import handle_command
from llmcli.config import Config
from llmcli.history import Conversation
from llmcli.memory import MemoryStore
from llmcli.ui import StreamingUI

PROFILES = """\
[defaults]
profile = "local"

[profiles.local]
provider = "ollama"
model = "qwen2.5-coder"
endpoint = "http://localhost:11434"

[profiles.remote]
provider = "openai_compat"
endpoint = "https://example.invalid/v1"
model = "remote-model"
api_key = "k"
max_tokens = 16384
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".llmcli").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    for key in list(__import__("os").environ):
        if key.startswith("LLMCLI_"):
            monkeypatch.delenv(key)
    return home


def write_profiles(home, text=PROFILES):
    path = home / ".llmcli" / "config.toml"
    path.write_text(text)
    path.chmod(0o600)


# --- display and commands ------------------------------------------------------

def make_ui():
    output = StringIO()
    return StreamingUI(Console(file=output, force_terminal=False, width=200)), output


class DummyProvider:
    model = "m"


def run(command, config, ui, switch=None, tmp_path=None):
    return handle_command(
        command,
        Conversation(system_prompt="", max_tokens=8192),
        config,
        DummyProvider(),
        ui,
        MemoryStore(root=tmp_path / "h", project_root=tmp_path / "p"),
        None,
        switch,
    )


def make_config(profile=None):
    return Config("ollama", "m", "http://localhost:11434", "", 8192, 4096, "CONTEXT.md", profile=profile)


def test_profiles_command_lists_profiles_and_key_status(home, tmp_path):
    write_profiles(home)
    ui, output = make_ui()
    run("/profiles", make_config("local"), ui, tmp_path=tmp_path)
    text = output.getvalue()
    assert "local" in text and "remote" in text
    assert "set" in text and "none" in text
    assert '"k"' not in text


def test_profiles_command_without_file_explains_built_in(home, tmp_path):
    ui, output = make_ui()
    run("/profiles", make_config(), ui, tmp_path=tmp_path)
    assert "built-in local Ollama" in output.getvalue()


def test_status_shows_config_source(home, tmp_path):
    ui, output = make_ui()
    run("/status", make_config("local"), ui, tmp_path=tmp_path)
    assert "Config:" in output.getvalue() and "built-in defaults" in output.getvalue()


def test_profile_command_dispatches_and_parses_keep(home, tmp_path):
    calls = []
    ui, _ = make_ui()
    run("/profile remote --keep", make_config(), ui, lambda n, k: calls.append((n, k)), tmp_path)
    run("/profile local", make_config(), ui, lambda n, k: calls.append((n, k)), tmp_path)
    assert calls == [("remote", True), ("local", False)]


def test_profile_command_without_name_shows_active(home, tmp_path):
    ui, output = make_ui()
    run("/profile", make_config("local"), ui, tmp_path=tmp_path)
    assert "local" in output.getvalue()


# --- conversation hand-over --------------------------------------------------------

def test_keep_text_only_drops_tool_traffic():
    convo = Conversation(system_prompt="", max_tokens=8192)
    convo.add("user", "read a.py")
    convo.add_message({"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]})
    convo.add_message({"role": "tool", "tool_call_id": "1", "content": "{}"})
    convo.add("assistant", "It prints hi.")
    dropped = convo.keep_text_only()
    assert dropped == 2
    assert convo.messages == [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": "It prints hi."},
    ]


# --- switching a live session ---------------------------------------------------------

class FakeProvider:
    def __init__(self, config, healthy=True):
        self.model = config.model
        self.healthy = healthy

    def check_health(self):
        return self.healthy, "ok" if self.healthy else "endpoint down"


class FakeAgent:
    provider = "original"


def switch_setup(home, monkeypatch, healthy=True):
    write_profiles(home)
    monkeypatch.setattr("llmcli.cli.create_provider", lambda cfg: FakeProvider(cfg, healthy))
    ui, output = make_ui()
    convo = Conversation(system_prompt="", max_tokens=8192)
    convo.add("user", "hello")
    convo.add("assistant", "hi there")
    return ui, output, convo, make_config("local"), FakeAgent()


def test_switch_replaces_provider_config_and_clears_history(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    ui, output, convo, config, agent = switch_setup(home, monkeypatch)
    new = switch_profile_session("remote", False, config, convo, agent, ui)
    assert new is agent.provider and new.model == "remote-model"
    assert config.profile == "remote" and config.provider == "openai_compat"
    assert config.api_key == "k" and convo.max_tokens == 8192
    assert convo.messages == []
    assert "Switched to profile" in output.getvalue()


def test_switch_with_keep_carries_text_over(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    ui, _, convo, config, agent = switch_setup(home, monkeypatch)
    switch_profile_session("remote", True, config, convo, agent, ui)
    assert [m["content"] for m in convo.messages] == ["hello", "hi there"]


def test_failed_health_check_changes_nothing(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    ui, output, convo, config, agent = switch_setup(home, monkeypatch, healthy=False)
    assert switch_profile_session("remote", False, config, convo, agent, ui) is None
    assert agent.provider == "original" and config.profile == "local"
    assert len(convo.messages) == 2
    assert "Not switched" in output.getvalue()


def test_unknown_profile_changes_nothing(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    ui, output, convo, config, agent = switch_setup(home, monkeypatch)
    assert switch_profile_session("nope", False, config, convo, agent, ui) is None
    assert config.profile == "local" and len(convo.messages) == 2
    assert "Available: local, remote" in output.getvalue()
