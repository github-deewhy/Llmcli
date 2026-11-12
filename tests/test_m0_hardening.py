"""M0 hardening: safe int parsing, one-shot errors, valid truncation, size caps."""

import json
from io import StringIO

import pytest
from rich.console import Console

from llmcli.cli import run_oneshot
from llmcli.coding_tools import MAX_TOOL_RESULT, ProjectCodingTools
from llmcli.config import Config, load_config
from llmcli.history import Conversation
from llmcli.memory import MemoryStore
from llmcli.providers.events import ProviderConnectError, ProviderHTTPError, TextDelta
from llmcli.providers.fake import ScriptedProvider
from llmcli.security import MAX_INSTRUCTION_BYTES, read_capped_text
from llmcli.ui import StreamingUI


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    import os
    for key in list(os.environ):
        if key.startswith("LLMCLI_"):
            monkeypatch.delenv(key)
    return home, project


# --- safe int parsing (now from config.toml) --------------------------------

@pytest.mark.parametrize("key", ["context_size", "max_tokens"])
@pytest.mark.parametrize("bad", ['"abc"', '"8k"', "0", "-5", "1.5"])
def test_bad_integer_settings_give_clear_error_not_traceback(isolated, key, bad):
    home, _ = isolated
    (home / ".llmcli").mkdir()
    (home / ".llmcli" / "config.toml").write_text(f"[defaults]\n{key} = {bad}\n")
    with pytest.raises(ValueError) as error:
        load_config({})
    assert key in str(error.value)


def test_valid_integer_settings_still_work(isolated):
    home, _ = isolated
    (home / ".llmcli").mkdir()
    (home / ".llmcli" / "config.toml").write_text("[defaults]\ncontext_size = 16384\n")
    assert load_config({}).context_size == 16384


# --- run_oneshot -----------------------------------------------------------

def FailingProvider(error):  # noqa: N802
    return ScriptedProvider([[error]])


def _oneshot(provider):
    out = StringIO()
    ui = StreamingUI(Console(file=out, force_terminal=False, width=100))
    config = Config("ollama", "m", "http://localhost:11434", "", 8192, 1024, "CONTEXT.md")
    code = run_oneshot(config, provider, Conversation(system_prompt="s", max_tokens=8192), ui, "hi")
    return code, out.getvalue()


def test_oneshot_connection_error_returns_1_with_short_message():
    error = ProviderConnectError("Cannot reach http://localhost:11434. Is the server running?")
    code, text = _oneshot(FailingProvider(error))
    assert code == 1
    assert "Cannot reach http://localhost:11434" in text
    assert "Traceback" not in text


def test_oneshot_http_status_error_mentions_status():
    code, text = _oneshot(FailingProvider(ProviderHTTPError("returned HTTP 401.", 401)))
    assert code == 1 and "401" in text


def test_oneshot_value_error_and_interrupt():
    assert _oneshot(FailingProvider(ValueError("bad reply")))[0] == 1
    assert _oneshot(FailingProvider(KeyboardInterrupt()))[0] == 130


def test_oneshot_success_returns_0():
    assert _oneshot(ScriptedProvider([[TextDelta("ok")]]))[0] == 0


# --- truncation keeps valid JSON and the useful budget ---------------------

def make_tools(tmp_path):
    return ProjectCodingTools(tmp_path, StreamingUI(Console(file=StringIO(), width=100)))


def test_large_string_result_is_valid_json_and_uses_most_of_the_budget(tmp_path):
    tools = make_tools(tmp_path)
    huge = {"output": "line\n" * 50_000}
    encoded = tools._fit(huge)
    data = json.loads(encoded)  # must parse
    assert len(encoded) <= MAX_TOOL_RESULT
    assert len(encoded) > MAX_TOOL_RESULT // 2  # old code kept only 1/8
    assert data["truncated"] is True
    assert "characters omitted" in data["output"]


def test_large_list_result_drops_items_and_reports_count(tmp_path):
    tools = make_tools(tmp_path)
    matches = [{"path": f"f{i}.py", "line": i, "text": "x" * 50} for i in range(2000)]
    encoded = tools._fit({"matches": matches})
    data = json.loads(encoded)
    assert len(encoded) <= MAX_TOOL_RESULT
    assert data["omitted_matches"] > 0
    assert len(data["matches"]) + data["omitted_matches"] == 2000


def test_small_result_is_untouched(tmp_path):
    tools = make_tools(tmp_path)
    assert json.loads(tools._fit({"a": 1})) == {"a": 1}


# --- size caps on memory and context files --------------------------------

def test_read_capped_text_truncates_with_notice(tmp_path):
    path = tmp_path / "big.md"
    path.write_text("a" * (MAX_INSTRUCTION_BYTES + 500), encoding="utf-8")
    text = read_capped_text(path)
    assert "truncated" in text and len(text) < MAX_INSTRUCTION_BYTES + 200


def test_read_capped_text_leaves_small_file_alone(tmp_path):
    path = tmp_path / "s.md"
    path.write_text("héllo", encoding="utf-8")
    assert read_capped_text(path) == "héllo"


def test_memory_read_is_capped_and_add_refuses_overflow(tmp_path):
    store = MemoryStore(root=tmp_path / "mem", project_root=tmp_path)
    store.paths["global"].parent.mkdir(parents=True)
    store.paths["global"].write_text("x" * (MAX_INSTRUCTION_BYTES + 10), encoding="utf-8")
    assert "truncated" in store.read()["global"]
    with pytest.raises(ValueError, match="would exceed"):
        store.add("global", "more")
