"""Tests for named profiles: parsing, precedence, key handling, and switching."""

from io import StringIO

import pytest
from rich.console import Console

from llmcli.cli import handle_command
from llmcli.config import Config, load_config
from llmcli.history import Conversation
from llmcli.memory import MemoryStore
from llmcli.profiles import load_profiles, summarize_profiles
from llmcli.ui import StreamingUI

PROFILES = """\
[profiles.local]
provider = "ollama"
model = "qwen2.5-coder"
endpoint = "http://localhost:11434"

[profiles.nvidia]
provider = "openai_compat"
endpoint = "https://integrate.api.nvidia.com/v1"
model = "nvidia/nemotron-3-ultra-550b-a55b"
api_key_env = "NVIDIA_API_KEY"
max_tokens = 16384
temperature = 1
top_p = 0.95
extra_body = { chat_template_kwargs = { enable_thinking = true } }
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
        if key.startswith("LLMCLI_") or key == "NVIDIA_API_KEY":
            monkeypatch.delenv(key)
    return home


def write_profiles(home, text=PROFILES):
    (home / ".llmcli" / "profiles.toml").write_text(text)


# --- parsing ---------------------------------------------------------------

def test_no_file_means_no_profiles(home):
    assert load_profiles() == (None, {})
    assert load_config().profile is None  # unchanged legacy behaviour


def test_profile_values_are_applied(home, monkeypatch):
    write_profiles(home)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    config = load_config(profile="nvidia")
    assert config.profile == "nvidia"
    assert config.provider == "openai_compat"
    assert config.endpoint == "https://integrate.api.nvidia.com/v1"
    assert config.max_tokens == 16384
    assert (config.temperature, config.top_p) == (1, 0.95)
    assert config.api_key == "nvapi-test"
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


@pytest.mark.parametrize(
    "text, message",
    [
        ('[profiles.a]\nmodle = "x"\n', "unknown setting"),
        ('[profiles.a]\nmax_tokens = "lots"\n', "max_tokens must be a positive integer"),
        ('[profiles.a]\napi_key = "sk-oops"\n', "unknown setting"),
        ('model = "x"\n', "unknown top-level"),
        ('default = "nope"\n[profiles.a]\nmodel = "x"\n', "not defined"),
        ('[profiles."bad name"]\nmodel = "x"\n', "Invalid profile name"),
        ("[profiles.a\n", "Could not read"),
    ],
)
def test_invalid_profiles_files_are_rejected(home, text, message):
    write_profiles(home, text)
    with pytest.raises(ValueError, match=message):
        load_profiles()


# --- selection and precedence ----------------------------------------------

def test_profile_selection_order(home, monkeypatch):
    write_profiles(home, 'default = "local"\n' + PROFILES)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    assert load_config().profile == "local"                       # file default
    monkeypatch.setenv("LLMCLI_PROFILE", "nvidia")
    assert load_config().profile == "nvidia"                      # env beats default
    assert load_config(profile="local").profile == "local"        # argument beats env


def test_unknown_profile_lists_choices(home):
    write_profiles(home)
    with pytest.raises(ValueError, match="Available: local, nvidia"):
        load_config(profile="nope")


def test_profile_beats_env_but_cli_beats_profile(home, monkeypatch):
    write_profiles(home)
    monkeypatch.setenv("LLMCLI_MODEL", "from-os-env")
    assert load_config(profile="local").model == "qwen2.5-coder"
    assert load_config({"model": "cli-model"}, profile="local").model == "cli-model"


def test_unset_profile_fields_fall_back_to_env(home, monkeypatch):
    write_profiles(home, '[profiles.min]\nprovider = "ollama"\n')
    monkeypatch.setenv("LLMCLI_MODEL", "from-os-env")
    assert load_config(profile="min").model == "from-os-env"


# --- API keys ----------------------------------------------------------------

def test_missing_key_env_is_a_clear_error(home):
    write_profiles(home)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY, which is not set"):
        load_config(profile="nvidia")


def test_key_can_come_from_dotenv_file(home):
    write_profiles(home)
    (home / ".llmcli" / ".env").write_text("NVIDIA_API_KEY=from-file\n")
    assert load_config(profile="nvidia").api_key == "from-file"


def test_profile_never_inherits_the_generic_api_key(home, monkeypatch):
    write_profiles(home)
    monkeypatch.setenv("LLMCLI_API_KEY", "generic-secret")
    assert load_config(profile="local").api_key == ""


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


def test_profiles_command_lists_profiles_and_key_status(home, tmp_path, monkeypatch):
    write_profiles(home)
    ui, output = make_ui()
    run("/profiles", make_config("local"), ui, tmp_path=tmp_path)
    text = output.getvalue()
    assert "local" in text and "nvidia" in text
    assert "NVIDIA_API_KEY missing" in text
    assert "none needed" in text


def test_profile_command_dispatches_and_parses_keep(home, tmp_path):
    calls = []
    ui, _ = make_ui()
    run("/profile nvidia --keep", make_config(), ui, lambda n, k: calls.append((n, k)), tmp_path)
    run("/profile local", make_config(), ui, lambda n, k: calls.append((n, k)), tmp_path)
    assert calls == [("nvidia", True), ("local", False)]


def test_profile_command_without_name_shows_active(home, tmp_path):
    ui, output = make_ui()
    run("/profile", make_config("local"), ui, tmp_path=tmp_path)
    assert "local" in output.getvalue()


def test_summarize_profiles_reports_key_state():
    rows = summarize_profiles({"a": {"api_key_env": "X"}, "b": {}}, lambda name: "v" if name == "X" else None)
    assert rows[0]["key_set"] is True and rows[1]["key_env"] is None


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
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    ui, output, convo, config, agent = switch_setup(home, monkeypatch)
    new = switch_profile_session("nvidia", False, config, convo, agent, ui)
    assert new is agent.provider and new.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.profile == "nvidia" and config.provider == "openai_compat"
    assert config.api_key == "k" and convo.max_tokens == 8192
    assert convo.messages == []
    assert "Switched to profile" in output.getvalue()


def test_switch_with_keep_carries_text_over(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    ui, _, convo, config, agent = switch_setup(home, monkeypatch)
    switch_profile_session("nvidia", True, config, convo, agent, ui)
    assert [m["content"] for m in convo.messages] == ["hello", "hi there"]


def test_failed_health_check_changes_nothing(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    ui, output, convo, config, agent = switch_setup(home, monkeypatch, healthy=False)
    assert switch_profile_session("nvidia", False, config, convo, agent, ui) is None
    assert agent.provider == "original" and config.profile == "local"
    assert len(convo.messages) == 2
    assert "Not switched" in output.getvalue()


def test_missing_key_changes_nothing(home, monkeypatch):
    from llmcli.cli import switch_profile_session
    ui, output, convo, config, agent = switch_setup(home, monkeypatch)
    assert switch_profile_session("nvidia", False, config, convo, agent, ui) is None
    assert config.profile == "local" and len(convo.messages) == 2
    assert "NVIDIA_API_KEY" in output.getvalue()
