"""Tests for interactive CLI commands."""

from io import StringIO

from rich.console import Console
from prompt_toolkit.styles import Style

from llmcli.cli import _confirm_memory_clear, build_keybindings, handle_command
from llmcli.config import Config
from llmcli.history import Conversation
from llmcli.memory import MemoryStore
from llmcli.ui import THEME_PRESETS, StreamingUI


class DummyProvider:
    model = "old-model"


def test_keybindings_use_supported_prompt_toolkit_keys():
    assert build_keybindings()


def test_memory_confirmation_restores_normal_prompt_message():
    class FakeSession:
        message = [("class:prompt", "❯ ")]

        def prompt(self, message):
            self.message = message
            return "clear"

    session = FakeSession()
    normal_message = session.message

    assert _confirm_memory_clear(session, "Type 'clear': ")
    assert session.message == normal_message


def make_config():
    return Config(
        provider="ollama",
        model="old-model",
        endpoint="http://localhost:11434",
        api_key="",
        context_size=8192,
        max_tokens=4096,
        context_file="CONTEXT.md",
    )


def test_help_command_displays_commands_and_keeps_repl_open(tmp_path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    memory_store = MemoryStore(root=tmp_path / "home", project_root=tmp_path / "project")

    should_continue = handle_command(
        "/help ",
        Conversation(system_prompt="", max_tokens=8192),
        make_config(),
        DummyProvider(),
        ui,
        memory_store,
    )

    assert should_continue
    assert "/model <name>" in output.getvalue()
    assert "/memory add [--global] <text>" in output.getvalue()
    assert "/memory clear [global|project|all]" in output.getvalue()
    assert "user-guide.md" in output.getvalue()
    assert "/quit, /exit" in output.getvalue()
    assert "Commands start with /" in output.getvalue()


def test_clear_with_extra_arguments_does_not_clear_conversation(tmp_path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    conversation = Conversation(system_prompt="", max_tokens=8192)
    conversation.add("user", "Keep this history.")
    memory_store = MemoryStore(root=tmp_path / "home", project_root=tmp_path / "project")

    handle_command(
        "/clear memory all",
        conversation,
        make_config(),
        DummyProvider(),
        ui,
        memory_store,
    )

    assert conversation.messages == [{"role": "user", "content": "Keep this history."}]
    assert "Usage: /clear" in output.getvalue()
    assert "To clear memory, use /memory clear" in output.getvalue()


def test_model_command_updates_provider_and_config(tmp_path):
    config = make_config()
    provider = DummyProvider()
    ui = StreamingUI(Console(file=StringIO(), force_terminal=False))
    memory_store = MemoryStore(root=tmp_path / "home", project_root=tmp_path / "project")

    handle_command(
        "/model new-model",
        Conversation(system_prompt="", max_tokens=8192),
        config,
        provider,
        ui,
        memory_store,
    )

    assert config.model == "new-model"
    assert provider.model == "new-model"


def test_theme_command_changes_theme_and_prompt_colors(tmp_path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    memory_store = MemoryStore(root=tmp_path / "home", project_root=tmp_path / "project")

    handle_command(
        "/theme sunset",
        Conversation(system_prompt="", max_tokens=8192),
        make_config(),
        DummyProvider(),
        ui,
        memory_store,
    )

    assert ui.theme == "sunset"
    assert ui.colors["accent"] == "bright_magenta"
    assert "Theme changed to sunset" in output.getvalue()


def test_each_theme_uses_a_valid_prompt_toolkit_color():
    ui = StreamingUI(Console(file=StringIO(), force_terminal=False))

    for theme in THEME_PRESETS:
        ui.set_theme(theme)
        Style.from_dict({"prompt": f"bold {ui.colors['prompt']}"})


def test_memory_commands_persist_and_reload_notes(tmp_path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    memory_store = MemoryStore(
        base_prompt="Coding instructions",
        root=tmp_path / "home",
        project_root=tmp_path / "project",
    )
    conversation = Conversation(
        system_prompt=memory_store.compose_system_prompt(),
        max_tokens=8192,
    )
    config = make_config()
    provider = DummyProvider()

    handle_command(
        "/memory add Prefer pytest",
        conversation,
        config,
        provider,
        ui,
        memory_store,
    )
    handle_command(
        "/memory add --global Use Python 3",
        conversation,
        config,
        provider,
        ui,
        memory_store,
    )

    assert "Prefer pytest" in memory_store.paths["project"].read_text()
    assert "Use Python 3" in memory_store.paths["global"].read_text()
    assert "Prefer pytest" in conversation.system_prompt
    assert "Use Python 3" in conversation.system_prompt

    handle_command(
        "/memory show",
        conversation,
        config,
        provider,
        ui,
        memory_store,
    )
    assert "Prefer pytest" in output.getvalue()
    assert "Use Python 3" in output.getvalue()

    memory_store.paths["project"].write_text("Prefer unittest", encoding="utf-8")
    handle_command(
        "/memory refresh",
        conversation,
        config,
        provider,
        ui,
        memory_store,
    )

    assert "Prefer unittest" in conversation.system_prompt


def test_memory_clear_all_requires_confirmation_and_updates_session(tmp_path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    memory_store = MemoryStore(
        base_prompt="Coding instructions",
        root=tmp_path / "home",
        project_root=tmp_path / "project",
    )
    memory_store.add("global", "Use concise answers.")
    memory_store.add("project", "Run tests with pytest.")
    conversation = Conversation(
        system_prompt=memory_store.compose_system_prompt(),
        max_tokens=8192,
    )

    confirmed_prompts = []
    handle_command(
        "/memory clear all",
        conversation,
        make_config(),
        DummyProvider(),
        ui,
        memory_store,
        lambda prompt: confirmed_prompts.append(prompt) or True,
    )

    assert confirmed_prompts == ["Type 'clear' to erase all memory: "]
    assert memory_store.read() == {"global": "", "project": ""}
    assert conversation.system_prompt == "Coding instructions"

    memory_store.add("project", "Keep this note.")
    handle_command(
        "/memory clear project",
        conversation,
        make_config(),
        DummyProvider(),
        ui,
        memory_store,
        lambda _prompt: False,
    )
    assert memory_store.read()["project"] == "Keep this note.\n"


def test_memory_without_action_shows_usage_and_paths(tmp_path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False, width=240))
    memory_store = MemoryStore(root=tmp_path / "home", project_root=tmp_path / "project")

    handle_command(
        "/memory",
        Conversation(system_prompt="", max_tokens=8192),
        make_config(),
        DummyProvider(),
        ui,
        memory_store,
    )

    assert "/memory show" in output.getvalue()
    assert "/memory clear" in output.getvalue()
    assert "[global|project|all]" in output.getvalue()
    assert str(memory_store.paths["global"]) in output.getvalue()
