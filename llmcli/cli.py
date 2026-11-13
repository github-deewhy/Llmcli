"""Main CLI entry point, REPL loop, and command routing."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import DynamicStyle, Style

from llmcli.agent import CodingAgent
from llmcli.budget import messages_chars, over_budget_warning, trim_notice
from llmcli.coding_tools import ProjectCodingTools
from llmcli.config import load_config
from llmcli.config_file import load_config_file, startup_messages, summarize_profiles
from llmcli.context import load_context
from llmcli.file_resolver import resolve_attachments
from llmcli.history import Conversation
from llmcli.memory import MemoryStore
from llmcli.providers import create_provider
from llmcli.providers.base import BaseProvider
from llmcli.providers.events import ProviderError
from llmcli.ui import THEME_PRESETS, StreamingUI

CODING_SYSTEM_PROMPT = (
    "You are a helpful coding CLI assistant. Focus on the user's code and explicit "
    "task; keep responses practical and avoid unrelated assumptions."
)


def build_keybindings() -> KeyBindings:
    """Use Enter to send and Escape+Enter to insert a newline."""
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    return bindings


def handle_command(
    cmd: str,
    conversation: Conversation,
    config,
    provider: BaseProvider,
    ui: StreamingUI,
    memory_store: MemoryStore,
    confirm: Callable[[str], bool] | None = None,
    switch_profile: Callable[[str, bool], None] | None = None,
    agent: CodingAgent | None = None,
) -> bool:
    """
    Handle slash commands. Returns True if the CLI should continue, False to exit.
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command in ["/quit", "/exit"]:
        ui.print_info("Goodbye!")
        return False
    if command in ["/help", "/?"]:
        ui.print_help()
    elif command in ["/clear", "/reset"]:
        if args:
            ui.print_error(
                "Usage: /clear (or /reset). To clear memory, use "
                "/memory clear with global, project, or all."
            )
        else:
            conversation.clear()
            ui.print_success("Conversation history cleared.")
    elif command in ["/config", "/status"]:
        ui.console.print(f"[dim]Config:[/dim] [bold]{getattr(config, 'source', 'built-in defaults')}[/bold]")
        ui.console.print(f"[dim]Profile:[/dim] [bold]{getattr(config, 'profile', None) or '(none)'}[/bold]")
        ui.console.print(f"[dim]Provider:[/dim] [bold]{config.provider}[/bold]")
        ui.console.print(f"[dim]Model:[/dim] [bold]{config.model}[/bold]")
        ui.console.print(f"[dim]Endpoint:[/dim] [bold]{config.endpoint}[/bold]")
        ui.console.print(f"[dim]Context Size:[/dim] [bold]{config.context_size}[/bold]")
        ui.console.print(f"[dim]Max Tokens:[/dim] [bold]{config.max_tokens}[/bold]")
        ui.console.print(f"[dim]Request timeout:[/dim] [bold]{getattr(config, 'request_timeout', 300)} s[/bold]")
        ui.console.print(f"[dim]Tool mode:[/dim] [bold]{_tool_mode_label(config, agent)}[/bold]")
        ui.console.print(f"[dim]Thinking text:[/dim] [bold]{'shown' if ui.show_reasoning else 'hidden'}[/bold]")
        ui.console.print(f"[dim]Theme:[/dim] [bold]{ui.theme}[/bold]")
        ui.print_capabilities(provider.capabilities, config.context_size)
        ui.print_usage(provider.last_usage)
    elif command in ["/system", "/context"]:
        if conversation.system_prompt:
            preview = conversation.system_prompt[:300] + ("..." if len(conversation.system_prompt) > 300 else "")
            ui.console.print(f"[dim cyan]System Context Loaded:[/dim cyan]\n{preview}")
        else:
            ui.print_info("No system context file loaded.")
    elif command == "/model":
        if not args:
            ui.print_error("Usage: /model <model_name>")
        else:
            config.model = args
            provider.model = args
            provider.reset_capabilities()
            provider.probe()
            ui.print_success(f"Model switched to: [bold]{args}[/bold]")
    elif command == "/profiles":
        try:
            config_file = load_config_file()
        except ValueError as error:
            ui.print_error(str(error))
        else:
            ui.print_profiles(summarize_profiles(config_file), getattr(config, "profile", None))
    elif command == "/profile":
        names = args.split()
        keep = "--keep" in names
        names = [name for name in names if name != "--keep"]
        if not names:
            ui.print_info(
                f"Active profile: [bold]{getattr(config, 'profile', None) or '(none)'}[/bold] · "
                r"switch with /profile <name> \[--keep] · list with /profiles"
            )
        elif len(names) > 1:
            ui.print_error(r"Usage: /profile <name> \[--keep]")
        elif switch_profile is None:
            ui.print_error("Profile switching is not available in this mode.")
        else:
            switch_profile(names[0], keep)
    elif command == "/models":
        with ui.show_spinner("Fetching models..."):
            models = provider.list_models()
        if models:
            ui.console.print("[dim]Available models:[/dim]")
            for model in models:
                ui.console.print(f"  • {model}")
        else:
            ui.print_error("Could not fetch models.")
    elif command == "/probe":
        if args:
            ui.print_error("Usage: /probe")
        else:
            try:
                with ui.show_spinner("Probing the model with one tiny tool call..."):
                    provider.probe()
                    provider.live_probe()
            except ProviderError as error:
                ui.print_error(f"Probe failed: {error}")
            except KeyboardInterrupt:
                ui.print_info("Probe cancelled.")
            else:
                ui.print_capabilities(provider.capabilities, config.context_size)
    elif command == "/thinking":
        choice = args.strip().lower()
        if choice in ("on", "off"):
            ui.show_reasoning = choice == "on"
            ui.print_success(f"Thinking text {'shown' if ui.show_reasoning else 'hidden'} for this session.")
        elif not choice:
            ui.print_info(
                f"Thinking text is {'shown' if ui.show_reasoning else 'hidden'}. "
                r"Change with /thinking on or /thinking off."
            )
        else:
            ui.print_error("Usage: /thinking [on|off]")
    elif command == "/theme":
        if not args:
            ui.print_info(f"Current theme: [bold]{ui.theme}[/bold]")
            ui.print_themes()
        else:
            try:
                ui.set_theme(args)
            except ValueError as error:
                ui.print_error(str(error))
            else:
                ui.print_success(f"Theme changed to [bold]{ui.theme}[/bold].")
    elif command == "/memory":
        _handle_memory_command(args, conversation, memory_store, ui, confirm)
    else:
        ui.print_error(f"Unknown command: {command}. Type /help for options.")

    return True


def _tool_mode_label(config, agent: CodingAgent | None) -> str:
    configured = getattr(config, "tool_mode", "auto")
    if agent is None or configured != "auto":
        return configured
    return f"auto (now: {agent.mode})"


def _handle_memory_command(
    args: str,
    conversation: Conversation,
    memory_store: MemoryStore,
    ui: StreamingUI,
    confirm: Callable[[str], bool] | None,
) -> None:
    parts = args.strip().split(maxsplit=1)
    if not parts:
        ui.print_info(
            r"Usage: /memory show \[global|project|all] · "
            r"/memory add \[--global] <text> · "
            r"/memory clear \[global|project|all] · /memory refresh"
        )
        ui.print_info("Memory files:")
        ui.print_memory_locations(memory_store.paths)
        return

    action = parts[0].lower()
    details = parts[1] if len(parts) > 1 else ""
    try:
        if action == "show":
            scope = details.lower() or "all"
            if scope == "all":
                scopes: tuple[str, ...] = ("global", "project")
            elif scope in memory_store.paths:
                scopes = (scope,)
            else:
                ui.print_error("Usage: /memory show [global|project|all]")
                return
            contents = memory_store.read()
            for selected_scope in scopes:
                ui.print_memory(
                    selected_scope,
                    str(memory_store.paths[selected_scope]),
                    contents[selected_scope],
                )
        elif action == "add":
            scope = "project"
            content = details
            if details == "--global":
                ui.print_error("Usage: /memory add [--global] <Markdown text>")
                return
            if details.startswith("--global "):
                scope = "global"
                content = details[len("--global "):]
            if not content.strip():
                ui.print_error("Usage: /memory add [--global] <Markdown text>")
                return
            memory_store.add(scope, content)
            conversation.system_prompt = memory_store.compose_system_prompt()
            ui.print_success(f"Added note to {scope} memory.")
        elif action == "refresh":
            if details:
                ui.print_error("Usage: /memory refresh")
                return
            conversation.system_prompt = memory_store.compose_system_prompt()
            ui.print_success("Memory reloaded for the current session.")
        elif action == "clear":
            scope = details.lower()
            if scope not in ("global", "project", "all"):
                ui.print_error("Usage: /memory clear [global|project|all]")
                return
            if confirm is None:
                ui.print_error("Memory clearing requires confirmation.")
                return
            if not confirm(f"Type 'clear' to erase {scope} memory: "):
                ui.print_info("Memory clear cancelled.")
                return
            clear_scopes = ("global", "project") if scope == "all" else (scope,)
            for selected_scope in clear_scopes:
                memory_store.clear(selected_scope)
            conversation.system_prompt = memory_store.compose_system_prompt()
            ui.print_success(f"Cleared {scope} memory.")
        else:
            ui.print_error(
                "Usage: /memory show [global|project|all] · "
                "/memory add [--global] <text> · "
                "/memory clear [global|project|all] · /memory refresh"
            )
    except (OSError, UnicodeError) as error:
        ui.print_error(f"Could not access memory file: {error}")


def _confirm_memory_clear(session: PromptSession, message: str) -> bool:
    return _prompt_confirmation(session, message, "clear")


def _prompt_confirmation(session: PromptSession, message: str, expected: str) -> bool:
    original_message = session.message
    try:
        answer = session.prompt(message=[("class:prompt", message)])
    except (EOFError, KeyboardInterrupt):
        return False
    finally:
        session.message = original_message
    return answer.strip().lower() == expected


def switch_profile_session(
    name: str,
    keep: bool,
    config,
    conversation: Conversation,
    agent: CodingAgent,
    ui: StreamingUI,
) -> BaseProvider | None:
    """
    Switch the running session to another profile.

    Returns the new provider on success. On any failure (unknown profile, missing
    API key, unreachable endpoint) an error is shown, nothing is changed, and
    None is returned.
    """
    try:
        new_config = load_config({"theme": ui.theme}, profile=name)
        new_provider = create_provider(new_config)
    except ValueError as error:
        ui.print_error(str(error))
        return None
    with ui.show_spinner(f"Checking {new_config.provider} connection..."):
        healthy, message = new_provider.check_health()
    if not healthy:
        ui.print_error(f"Not switched to '{name}': {message}")
        return None

    new_provider.probe()
    old_provider = agent.provider
    config.__dict__.update(vars(new_config))
    agent.set_provider(new_provider, new_config.tool_mode)
    agent.apply_settings(new_config.edit_matching, new_config.edit_tools, new_config.stop_guard)
    old_provider.close()
    conversation.max_tokens = new_config.context_size
    conversation.reply_tokens = new_config.max_tokens
    conversation.estimator.reset()
    if keep:
        dropped = conversation.keep_text_only()
        note = f"kept the conversation text (dropped {dropped} tool/empty messages)"
    else:
        conversation.clear()
        note = "started a fresh conversation (use --keep to carry text over)"
    ui.print_success(
        f"Switched to profile [bold]{name}[/bold] · {new_config.provider} / "
        f"[bold]{new_config.model}[/bold] · {note}"
    )
    return new_provider


def run_repl(config, provider, conversation: Conversation, ui: StreamingUI, memory_store: MemoryStore):
    """Run the interactive REPL loop."""
    history_file = Path.home() / ".llmcli" / "history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_file)),
        multiline=True,
        key_bindings=build_keybindings(),
        message=[("class:prompt", "❯ ")],
        style=DynamicStyle(
            lambda: Style.from_dict({"prompt": f"bold {ui.colors['prompt']}"})
        ),
    )

    ui.print_info(
        f"Ready · [bold]{config.provider}[/bold] / [bold]{config.model}[/bold] "
        "· agent tools on · edits/tests require approval · Enter to send · /help"
    )

    def confirm_memory_clear(message: str) -> bool:
        return _confirm_memory_clear(session, message)

    def confirm_agent_action(message: str) -> bool:
        return _prompt_confirmation(session, message, "yes")

    tools = ProjectCodingTools(Path.cwd(), ui)
    agent = CodingAgent(
        provider, conversation, tools, ui, confirm_agent_action, getattr(config, "tool_mode", "auto")
    )
    agent.apply_settings(
        getattr(config, "edit_matching", "exact"),
        getattr(config, "edit_tools", "replace"),
        getattr(config, "stop_guard", True),
    )

    def switch_profile(name: str, keep: bool) -> None:
        nonlocal provider
        new_provider = switch_profile_session(name, keep, config, conversation, agent, ui)
        if new_provider is not None:
            provider = new_provider

    while True:
        try:
            user_input = session.prompt()
        except (EOFError, KeyboardInterrupt):
            ui.print_info("\nGoodbye!")
            break

        if not user_input.strip():
            continue

        stripped_input = user_input.strip()
        if stripped_input.startswith("/"):
            if not handle_command(
                stripped_input,
                conversation,
                config,
                provider,
                ui,
                memory_store,
                confirm_memory_clear,
                switch_profile,
                agent,
            ):
                break
            continue

        resolved_input = resolve_attachments(user_input)
        agent.run(resolved_input)
        ui.console.print()


def run_oneshot(config, provider, conversation: Conversation, ui: StreamingUI, prompt: str) -> int:
    """Run a single prompt and exit (useful for scripting/SSH).

    Returns a process exit code: 0 on success, 1 on a provider or I/O error,
    130 when interrupted. Errors are printed as short messages, never tracebacks.
    """
    try:
        resolved_input = resolve_attachments(prompt)
        conversation.add("user", resolved_input)
        trim = conversation.fit_for_chat()
        notice = trim_notice(trim, conversation.max_tokens)
        if notice:
            ui.print_info(notice)
        warning = over_budget_warning(trim, conversation.max_tokens)
        if warning:
            ui.print_error(warning)
        messages = conversation.get_messages_for_api()
        result = ui.render_events(provider.stream(messages))
        if result.usage is not None:
            conversation.estimator.calibrate(messages_chars(messages), result.usage.prompt_tokens)
        if result.interrupted:
            return 130
        conversation.add("assistant", result.text)
    except KeyboardInterrupt:
        ui.print_error("Interrupted.")
        return 130
    except ProviderError as error:
        ui.print_error(str(error))
        return 1
    except (ValueError, OSError, UnicodeError) as error:
        ui.print_error(str(error))
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="LLM CLI - Chat with local or public models")
    parser.add_argument("-p", "--prompt", type=str, help="One-shot prompt (exits after response)")
    parser.add_argument("--model", type=str, help="Override the model specified in config")
    parser.add_argument(
        "--theme",
        choices=tuple(THEME_PRESETS),
        help="CLI color preset (overrides theme in ~/.llmcli/config.toml)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Named profile from ~/.llmcli/config.toml (overrides profile in [defaults])",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List the profiles in ~/.llmcli/config.toml and exit",
    )
    args = parser.parse_args()

    try:
        config_file = load_config_file()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(2)
    for message in startup_messages(config_file):
        print(message, file=sys.stderr)

    if args.list_profiles:
        list_ui = StreamingUI()
        default = config_file.defaults.get("profile")
        list_ui.print_profiles(summarize_profiles(config_file), default)
        return

    cli_overrides = {}
    if args.model:
        cli_overrides["model"] = args.model
    if args.theme:
        cli_overrides["theme"] = args.theme

    try:
        config = load_config(cli_overrides, profile=args.profile, file=config_file)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(2)
    ui = StreamingUI()
    try:
        ui.set_theme(config.theme)
    except ValueError as error:
        ui.print_error(str(error))
        sys.exit(2)
    ui.print_banner(config.provider, config.model)
    if config.profile:
        ui.print_info(f"Profile: [bold]{config.profile}[/bold]")

    try:
        provider = create_provider(config)
    except ValueError as e:
        ui.print_error(str(e))
        sys.exit(1)

    with ui.show_spinner(f"Checking {config.provider} connection..."):
        is_healthy, message = provider.check_health()

    if not is_healthy:
        ui.print_error(message)
        sys.exit(1)
    ui.print_success(message)
    provider.probe()

    context_prompt = load_context(config)
    base_system_prompt = "\n\n".join(
        part for part in (CODING_SYSTEM_PROMPT, context_prompt) if part
    )
    memory_store = MemoryStore(base_prompt=base_system_prompt)
    try:
        loaded_memory = memory_store.read()
        conversation = Conversation(
            system_prompt=memory_store.compose_system_prompt(loaded_memory),
            max_tokens=config.context_size,
            reply_tokens=config.max_tokens,
        )
    except (OSError, UnicodeError) as error:
        ui.print_error(f"Could not load local memory: {error}")
        sys.exit(1)

    if context_prompt:
        ui.print_info(f"Loaded context from: {config.context_file}")
    if any(loaded_memory.values()):
        ui.print_info("Loaded local persistent memory.")

    if args.prompt:
        if args.prompt == "-":
            prompt_text = sys.stdin.read()
        else:
            prompt_text = args.prompt
        sys.exit(run_oneshot(config, provider, conversation, ui, prompt_text))
    else:
        run_repl(config, provider, conversation, ui, memory_store)


if __name__ == "__main__":
    main()
