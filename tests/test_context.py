"""Tests for context file loading."""

import os
import tempfile
from pathlib import Path

from llmcli.config import Config
from llmcli.context import load_context
from llmcli.history import Conversation


def test_load_context_cwd_priority():
    """Test that CWD context file is preferred over home directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd_file = Path(tmpdir) / "CONTEXT.md"
        cwd_file.write_text("CWD Context")

        home_dir = Path(tmpdir) / "home"
        home_dir.mkdir()
        home_file = home_dir / "CONTEXT.md"
        home_file.write_text("Home Context")

        config = Config(
            provider="ollama",
            model="test",
            endpoint="http://localhost",
            api_key="",
            context_size=8192,
            max_tokens=4096,
            context_file="CONTEXT.md",
        )

        original_cwd = Path.cwd()
        original_home = Path.home

        Path.home = lambda: home_dir
        os.chdir(tmpdir)
        try:
            content = load_context(config)
            assert content == "CWD Context"
        finally:
            os.chdir(original_cwd)
            Path.home = original_home


def test_load_context_fallback_to_home():
    """Test that home directory context is loaded if CWD has none."""
    with tempfile.TemporaryDirectory() as tmpdir:
        home_dir = Path(tmpdir) / "home"
        home_dir.mkdir()
        home_file = home_dir / "CONTEXT.md"
        home_file.write_text("Home Context")

        config = Config(
            provider="ollama",
            model="test",
            endpoint="http://localhost",
            api_key="",
            context_size=8192,
            max_tokens=4096,
            context_file="CONTEXT.md",
        )

        original_home = Path.home
        Path.home = lambda: home_dir

        with tempfile.TemporaryDirectory() as empty_cwd:
            original_cwd = Path.cwd()
            os.chdir(empty_cwd)
            try:
                content = load_context(config)
                assert content == "Home Context"
            finally:
                os.chdir(original_cwd)
                Path.home = original_home


def test_load_context_missing_file():
    """Test that missing context files return empty string without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            provider="ollama",
            model="test",
            endpoint="http://localhost",
            api_key="",
            context_size=8192,
            max_tokens=4096,
            context_file="MISSING.md",
        )

        original_cwd = Path.cwd()
        os.chdir(tmpdir)
        try:
            content = load_context(config)
            assert content == ""
        finally:
            os.chdir(original_cwd)


def test_conversation_trimming_keeps_tool_messages_with_their_turn():
    conversation = Conversation(system_prompt="", max_tokens=8)
    conversation.add("user", "old question")
    conversation.add_message({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "function": {"name": "read_file", "arguments": {}}}],
    })
    conversation.add_message({"role": "tool", "tool_call_id": "call-1", "content": "contents"})
    conversation.add("assistant", "old answer")
    conversation.add("user", "new question")
    conversation.fit(0)  # trimming happens right before a request (spec 16, C1)

    assert [message["role"] for message in conversation.messages] == ["user"]
    assert conversation.messages[0]["content"] == "new question"


def test_agents_md_is_default_and_context_md_is_legacy_alias(tmp_path, monkeypatch):
    from llmcli.config import Config
    from llmcli.context import load_context

    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    config = Config("ollama", "m", "http://localhost:11434", "", 8192, 1024, "AGENTS.md")

    (project / "CONTEXT.md").write_text("legacy", encoding="utf-8")
    assert load_context(config) == "legacy"

    (project / "AGENTS.md").write_text("primary", encoding="utf-8")
    assert load_context(config) == "primary"
