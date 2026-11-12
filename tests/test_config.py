"""Single configuration file (D-018, spec 13): precedence, selection, validation, warnings."""

import os

import pytest

from llmcli.config import BUILT_IN_SOURCE, load_config
from llmcli.config_file import (
    key_permission_warning,
    legacy_notice,
    load_config_file,
    summarize_profiles,
)

SECRET = "sk-test-SECRET-value"

TWO_PROFILES = f"""\
[defaults]
profile = "ollama"
theme = "forest"
context_size = 16384
temperature = 0.2

[profiles.ollama]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "qwen2.5-coder"

[profiles.remote]
provider = "openai_compat"
endpoint = "https://example.invalid/v1/"
model = "remote-model"
api_key = "{SECRET}"
context_size = 32768
extra_body = {{ chat_template_kwargs = {{ enable_thinking = true }} }}
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Empty HOME and working directory, no LLMCLI_* variables."""
    home = tmp_path / "home"
    (home / ".llmcli").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    for key in list(os.environ):
        if key.startswith("LLMCLI_"):
            monkeypatch.delenv(key)
    return home


def write(home, text):
    path = home / ".llmcli" / "config.toml"
    path.write_text(text)
    path.chmod(0o600)
    return path


# --- 1. no file -------------------------------------------------------------

def test_no_file_gives_built_in_local_ollama(home):
    config = load_config()
    assert (config.provider, config.endpoint, config.model) == (
        "ollama", "http://localhost:11434", "qwen2.5-coder")
    assert config.api_key == "" and config.profile is None
    assert (config.context_size, config.max_tokens) == (8192, 4096)
    assert (config.theme, config.context_file) == ("ocean", "AGENTS.md")
    assert config.source == BUILT_IN_SOURCE


def test_file_with_only_defaults_uses_built_in_connection(home):
    write(home, '[defaults]\ntheme = "forest"\n')
    config = load_config()
    assert config.provider == "ollama" and config.theme == "forest"
    assert config.source == "~/.llmcli/config.toml"


# --- 2. precedence ----------------------------------------------------------

def test_defaults_beat_built_in(home):
    write(home, TWO_PROFILES)
    config = load_config()
    assert config.profile == "ollama"
    assert config.context_size == 16384 and config.temperature == 0.2
    assert config.max_tokens == 4096  # built-in, set nowhere else


def test_profile_beats_defaults(home):
    write(home, TWO_PROFILES)
    config = load_config(profile="remote")
    assert config.context_size == 32768          # profile
    assert config.temperature == 0.2             # [defaults]
    assert config.endpoint == "https://example.invalid/v1"
    assert config.api_key == SECRET
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_cli_beats_profile(home):
    write(home, TWO_PROFILES)
    config = load_config({"model": "cli-model", "theme": "sunset"}, profile="remote")
    assert config.model == "cli-model" and config.theme == "sunset"


def test_profile_extra_body_replaces_default(home):
    write(home, """\
[defaults]
extra_body = { a = 1, b = 2 }
[profiles.x]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "m"
extra_body = { c = 3 }
""")
    assert load_config().extra_body == {"c": 3}


# --- 3. profile selection ---------------------------------------------------

def test_explicit_profile_beats_defaults_profile(home):
    write(home, TWO_PROFILES)
    assert load_config(profile="remote").profile == "remote"


def test_single_profile_is_used_without_default(home):
    write(home, '[profiles.only]\nprovider = "ollama"\nendpoint = "http://h:1"\nmodel = "m"\n')
    assert load_config().profile == "only"


def test_several_profiles_without_default_is_an_error(home):
    write(home, TWO_PROFILES.replace('profile = "ollama"\n', ""))
    with pytest.raises(ValueError, match="Several profiles.*ollama, remote.*--profile"):
        load_config()


def test_unknown_profile_lists_choices(home):
    write(home, TWO_PROFILES)
    with pytest.raises(ValueError, match="Available: ollama, remote"):
        load_config(profile="nope")


def test_unknown_profile_without_any_profiles(home):
    with pytest.raises(ValueError, match="No profiles are defined"):
        load_config(profile="nope")


# --- 4 and 5. environment and .env files are ignored ------------------------

def test_environment_variables_are_ignored(home, monkeypatch):
    write(home, TWO_PROFILES)
    monkeypatch.setenv("LLMCLI_MODEL", "from-env")
    monkeypatch.setenv("LLMCLI_PROFILE", "remote")
    monkeypatch.setenv("LLMCLI_API_KEY", "env-key")
    monkeypatch.setenv("LLMCLI_ENDPOINT", "http://attacker.invalid")
    config = load_config()
    assert config.profile == "ollama" and config.model == "qwen2.5-coder"
    assert config.api_key == "" and config.endpoint == "http://localhost:11434"


def test_dotenv_files_are_ignored(home, tmp_path):
    lines = "LLMCLI_ENDPOINT=http://attacker.invalid\nLLMCLI_API_KEY=leak\nLLMCLI_MODEL=x\n"
    (home / ".llmcli" / ".env").write_text(lines)
    (tmp_path / "project" / ".env").write_text(lines)
    config = load_config()
    assert config.endpoint == "http://localhost:11434"
    assert config.api_key == "" and config.model == "qwen2.5-coder"


def test_loading_does_not_mutate_environment(home):
    (home / ".llmcli" / ".env").write_text("LLMCLI_MODEL=x\n")
    load_config()
    assert "LLMCLI_MODEL" not in os.environ


# --- 6. placement rules -----------------------------------------------------

@pytest.mark.parametrize("line", [f'api_key = "{SECRET}"', 'endpoint = "http://h:1"',
                                  'provider = "ollama"', 'model = "m"'])
def test_connection_settings_rejected_in_defaults(home, line):
    write(home, f"[defaults]\n{line}\n")
    with pytest.raises(ValueError, match=r"\[defaults\].*not allowed here.*profiles") as error:
        load_config_file()
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("line", ['theme = "forest"', 'context_file = "X.md"', 'profile = "p"'])
def test_defaults_only_settings_rejected_in_profile(home, line):
    write(home, f'[profiles.p]\nprovider = "ollama"\nendpoint = "http://h:1"\nmodel = "m"\n{line}\n')
    with pytest.raises(ValueError, match=r"not allowed here; put it in \[defaults\]"):
        load_config_file()


# --- 7. validation ------------------------------------------------------------

PROFILE_HEAD = '[profiles.p]\nprovider = "ollama"\nendpoint = "http://h:1"\nmodel = "m"\n'


@pytest.mark.parametrize(
    "text, message",
    [
        ('[profiles.p]\nprovider = "ollama"\nmodel = "m"\n', "missing required setting.*endpoint"),
        ('[profiles.p]\nprovider = "magic"\nendpoint = "http://h:1"\nmodel = "m"\n', "provider must be one of"),
        ('[profiles.p]\nprovider = "ollama"\nendpoint = "localhost:11434"\nmodel = "m"\n', "endpoint must be an http"),
        ('[profiles.p]\nprovider = "ollama"\nendpoint = "ftp://h"\nmodel = "m"\n', "endpoint must be an http"),
        (PROFILE_HEAD + "max_tokens = 0\n", "max_tokens must be a whole number"),
        (PROFILE_HEAD + 'context_size = "8k"\n', "context_size must be a whole number"),
        (PROFILE_HEAD + "context_size = true\n", "context_size must be a whole number"),
        (PROFILE_HEAD + 'temperature = "hot"\n', "temperature must be a number"),
        (PROFILE_HEAD + 'extra_body = "{}"\n', "extra_body must be a table"),
        (PROFILE_HEAD + 'modle = "x"\n', "unknown setting 'modle'"),
        ('[profiles."bad name"]\nprovider = "ollama"\nendpoint = "http://h:1"\nmodel = "m"\n', "invalid profile name"),
        ('model = "x"\n', "unknown top-level entry 'model'"),
        ('default = "p"\n' + PROFILE_HEAD, r'profile = "<name>" inside \[defaults\]'),
        ('[defaults]\nprofile = "nope"\n' + PROFILE_HEAD, "profile 'nope' is not defined"),
        ("[profiles.p\n", "not valid TOML.*line 1"),
    ],
)
def test_invalid_files_give_one_line_errors(home, text, message):
    write(home, text)
    with pytest.raises(ValueError, match=message) as error:
        load_config()
    assert "\n" not in str(error.value)
    assert str(error.value).startswith("~/.llmcli/config.toml")


def test_api_key_must_be_a_string(home):
    write(home, PROFILE_HEAD + "api_key = 123\n")
    with pytest.raises(ValueError, match="api_key must be a string"):
        load_config()


def test_errors_never_contain_the_key(home):
    write(home, PROFILE_HEAD + f'api_key = "{SECRET}"\nmax_tokens = -1\n')
    with pytest.raises(ValueError) as error:
        load_config()
    assert SECRET not in str(error.value)


def test_toml_error_does_not_echo_the_key(home):
    write(home, f'[profiles.p]\napi_key = "{SECRET}" oops\n')
    with pytest.raises(ValueError) as error:
        load_config()
    assert SECRET not in str(error.value)


def test_summary_reports_key_state_without_the_key(home):
    write(home, TWO_PROFILES)
    rows = summarize_profiles(load_config_file())
    assert [(r["name"], r["key_set"]) for r in rows] == [("ollama", False), ("remote", True)]
    assert SECRET not in repr(rows)


# --- 8. key file permissions ----------------------------------------------------

def test_permission_warning_for_readable_file_with_key(home):
    path = write(home, TWO_PROFILES)
    path.chmod(0o644)
    warning = key_permission_warning(load_config_file())
    assert warning and "chmod 600 ~/.llmcli/config.toml" in warning
    assert SECRET not in warning


def test_no_permission_warning_for_private_file(home):
    write(home, TWO_PROFILES)  # 0o600
    assert key_permission_warning(load_config_file()) is None


def test_no_permission_warning_without_a_key(home):
    path = write(home, TWO_PROFILES.replace(f'api_key = "{SECRET}"\n', ""))
    path.chmod(0o644)
    assert key_permission_warning(load_config_file()) is None


def test_empty_key_counts_as_no_key(home):
    path = write(home, TWO_PROFILES.replace(SECRET, ""))
    path.chmod(0o644)
    assert key_permission_warning(load_config_file()) is None
    assert load_config(profile="remote").api_key == ""


# --- 9. legacy notice --------------------------------------------------------------

def test_no_legacy_notice_on_a_clean_setup(home):
    assert legacy_notice(environ_names=[]) is None


@pytest.mark.parametrize("name", ["profiles.toml", ".env"])
def test_legacy_notice_for_old_home_files(home, name):
    (home / ".llmcli" / name).write_text("x")
    notice = legacy_notice(environ_names=[])
    assert notice and f"~/.llmcli/{name}" in notice and "config.toml" in notice


def test_legacy_notice_names_env_variables_but_not_values(home, monkeypatch):
    monkeypatch.setenv("LLMCLI_API_KEY", SECRET)
    notice = legacy_notice()
    assert notice and "LLMCLI_API_KEY" in notice and SECRET not in notice


def test_project_dotenv_does_not_trigger_notice(home, tmp_path):
    (tmp_path / "project" / ".env").write_text("LLMCLI_MODEL=x\n")
    assert legacy_notice(environ_names=[]) is None
