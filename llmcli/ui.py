"""Terminal UI using Rich for a polished, throttled coding CLI."""

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from llmcli.providers.base_types import ToolCall
from llmcli.providers.events import (
    Capabilities,
    Done,
    ReasoningDelta,
    TextDelta,
    ToolCallEvent,
    Usage,
)


@dataclass
class TurnResult:
    """Everything one streamed reply produced (spec 14, section 2.8)."""

    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    interrupted: bool = False


THEME_PRESETS = {
    "ocean": {
        "accent": "bright_cyan",
        "border": "bright_blue",
        "success": "bright_green",
        "error": "bright_red",
        "prompt": "ansibrightcyan",
    },
    "forest": {
        "accent": "bright_green",
        "border": "green",
        "success": "bright_green",
        "error": "bright_red",
        "prompt": "ansibrightgreen",
    },
    "sunset": {
        "accent": "bright_magenta",
        "border": "magenta",
        "success": "bright_yellow",
        "error": "bright_red",
        "prompt": "ansibrightmagenta",
    },
    "mono": {
        "accent": "bright_white",
        "border": "white",
        "success": "bright_white",
        "error": "bright_white",
        "prompt": "ansiwhite",
    },
}


class StreamingUI:
    def __init__(self, console: Console | None = None, theme: str = "ocean"):
        self.console = console or Console()
        self.update_interval = 0.125  # ~8 FPS to prevent flickering
        # Show the model's thinking text (``/thinking on``); off by default (P7).
        self.show_reasoning = False
        self.set_theme(theme)

    @property
    def colors(self) -> dict[str, str]:
        return THEME_PRESETS[self.theme]

    def set_theme(self, theme: str) -> None:
        theme = theme.lower()
        if theme not in THEME_PRESETS:
            options = ", ".join(THEME_PRESETS)
            raise ValueError(f"Unknown theme '{theme}'. Choose from: {options}.")
        self.theme = theme

    def print_banner(self, provider: str, model: str) -> None:
        colors = self.colors
        details = Table.grid(padding=(0, 2))
        details.add_column(style=f"bold {colors['accent']}", no_wrap=True)
        details.add_column()
        details.add_row("◈ LLMCLI", "[bold]CODING CONSOLE[/bold]")
        details.add_row("", f"[dim]{provider}[/dim]  ·  [bold]{model}[/bold]")
        self.console.print(
            Panel(
                details,
                title="[bold white]code, with context[/bold white]",
                subtitle="[dim]Enter to send  ·  Esc+Enter for newline  ·  /help[/dim]",
                border_style=colors["border"],
                padding=(0, 2),
            )
        )

    def print_help(self) -> None:
        commands = Table(
            show_header=True,
            header_style=f"bold {self.colors['accent']}",
            border_style="dim",
            pad_edge=False,
        )
        commands.add_column("Command", style="bold", no_wrap=True)
        commands.add_column("What it does")
        for command, description in (
            ("/help, /?", "Show available commands"),
            ("/clear, /reset", "Clear the conversation"),
            ("/status, /config", "Show settings, capabilities and last-turn speed"),
            ("/system, /context", "Show the loaded coding instructions"),
            (r"/profiles", "List configured provider profiles"),
            (r"/profile \[name] \[--keep]", "Show the active profile, or switch to another"),
            ("/model <name>", "Switch the model within the current profile"),
            ("/models", "List models available from the provider"),
            ("/probe", "Test the model with one tiny tool call and show its capabilities"),
            (r"/thinking \[on|off]", "Show or hide the model's thinking text"),
            (r"/theme \[name]", "Show or change the color preset"),
            ("/memory", "Show memory commands and file locations"),
            (r"/memory show \[global|project|all]", "Display selected Markdown memory"),
            (r"/memory add \[--global] <text>", "Append a Markdown memory note"),
            (r"/memory clear \[global|project|all]", "Clear selected memory after confirmation"),
            ("/memory refresh", "Reload memory files into the current session"),
            ("/quit, /exit", "Leave the CLI"),
        ):
            commands.add_row(command, description)
        self.console.print(
            Panel(commands, title="[bold]Command palette[/bold]", border_style=self.colors["border"])
        )
        self.console.print("[dim]Full guide: user-guide.md · CLI options: llmcli --help[/dim]")
        self.console.print(
            "[dim]Agent tools inspect project files and logs; edits and test runs require approval.[/dim]"
        )
        self.console.print(
            "[dim]Commands start with /. Ctrl-C cancels generation; at the prompt it exits.[/dim]"
        )

    def print_info(self, message: str) -> None:
        self.console.print(f"[{self.colors['accent']}]· {message}[/{self.colors['accent']}]")

    def print_success(self, message: str) -> None:
        self.console.print(f"[{self.colors['success']}]✓ {message}[/{self.colors['success']}]")

    def print_error(self, message: str) -> None:
        self.console.print(f"[{self.colors['error']}]✗ {message}[/{self.colors['error']}]")

    def print_profiles(self, rows: list[dict], active: str | None) -> None:
        """Show configured profiles; each row has name/provider/model/endpoint/key_set."""
        if not rows:
            self.print_info(
                "No profiles defined in ~/.llmcli/config.toml; using the built-in local Ollama. "
                "See config.toml.example in the project folder."
            )
            return
        table = Table(
            show_header=True,
            header_style=f"bold {self.colors['accent']}",
            border_style="dim",
            pad_edge=False,
        )
        table.add_column("Profile", no_wrap=True)
        for column in ("Provider", "Model", "Endpoint", "API key"):
            table.add_column(column, overflow="fold")
        for row in rows:
            key = "[green]set[/green]" if row["key_set"] else "[dim]none[/dim]"
            marker = "● " if row["name"] == active else ""
            table.add_row(
                f"{marker}[bold]{escape(row['name'])}[/bold]",
                escape(row["provider"]),
                escape(row["model"]),
                escape(row["endpoint"]),
                key,
            )
        self.console.print(table)

    def print_themes(self) -> None:
        themes = Table(show_header=True, header_style=f"bold {self.colors['accent']}", pad_edge=False)
        themes.add_column("Preset", style="bold")
        themes.add_column("Palette")
        themes.add_column("Use")
        for name, palette, usage in (
            ("ocean", "Cyan + blue", "Default, balanced contrast"),
            ("forest", "Green", "Calm, high-visibility accent"),
            ("sunset", "Magenta + amber", "Warm, vivid highlights"),
            ("mono", "White + gray", "Monochrome terminals"),
        ):
            themes.add_row(name, palette, usage)
        self.console.print(Panel(themes, title="Color presets", border_style=self.colors["border"]))

    def print_memory(self, scope: str, path: str, content: str) -> None:
        body = Markdown(content) if content.strip() else Text("No notes saved yet.", style="dim")
        self.console.print(Text.assemble((f"{scope.title()} memory ", "bold"), (path, "dim")))
        self.console.print(Panel(body, border_style=self.colors["border"]))

    def print_memory_locations(self, paths: dict[str, Path]) -> None:
        for scope, path in paths.items():
            self.console.print(
                Text.assemble((f"  {scope.title()}: ", "bold"), (str(path), "dim"))
            )

    def print_diff(self, path: str, diff: str) -> None:
        self.console.print(
            Panel(
                Syntax(diff, "diff", word_wrap=True),
                title=f"Proposed edit · {path}",
                border_style=self.colors["border"],
            )
        )

    def print_test_result(self, command: str, status: str, output: str) -> None:
        color = self.colors["success"] if status == "passed" else self.colors["error"]
        body = Syntax(output or "No output.", "text", word_wrap=True)
        self.console.print(
            Panel(
                body,
                title=f"{status.upper()} · {command}",
                border_style=color,
            )
        )

    def show_spinner(self, message: str):
        """Context manager for loading spinners."""
        return self.console.status(f"[cyan]{message}[/cyan]", spinner="dots")

    def render_stream(self, chunk_iterator: Iterable[str]) -> str:
        """
        Render a live response in a terminal and a single final frame when piped.
        """
        buffer = ""
        last_update = 0.0

        panel = Panel(Markdown(""), title="LLMCLI · response", border_style=self.colors["border"])

        if not self.console.is_terminal:
            buffer = "".join(chunk_iterator)
            panel.renderable = Markdown(buffer or "[dim]No response received.[/dim]")
            self.console.print(panel)
            return buffer

        try:
            with Live(
                panel,
                console=self.console,
                transient=True,
                vertical_overflow="ellipsis",
                refresh_per_second=8,
            ) as live:
                for chunk in chunk_iterator:
                    buffer += chunk
                    now = time.monotonic()
                    if now - last_update > self.update_interval:
                        panel.renderable = Markdown(buffer)
                        live.update(panel)
                        last_update = now
                panel.renderable = Markdown(buffer or "[dim]No response received.[/dim]")
                live.update(panel, refresh=True)
        except KeyboardInterrupt:
            buffer += "\n\n*[Generation interrupted by user]*"
            panel.renderable = Markdown(buffer)
            panel.title = "LLMCLI · interrupted"
            self.print_info("Generation cancelled.")

        self.console.print(panel)
        return buffer

    # --- streamed events (spec 14) ---------------------------------------------

    def _event_view(self, result: TurnResult, title: str):
        parts: list[Any] = []
        if result.reasoning:
            if self.show_reasoning:
                parts.append(Text(result.reasoning, style="dim italic"))
            elif not result.text:
                parts.append(Text(f"thinking… ({len(result.reasoning)} characters)", style="dim italic"))
        if result.text:
            parts.append(Markdown(result.text))
        if not parts:
            parts.append(Text("…", style="dim"))
        return Panel(Group(*parts), title=title, border_style=self.colors["border"])

    def render_events(
        self,
        events: Iterator[Any],
        title: str = "LLMCLI · response",
        show_empty: bool = True,
    ) -> TurnResult:
        """Stream events to the terminal and collect them.

        Ctrl-C closes the stream at once and returns a result marked ``interrupted``.
        ``ProviderError`` from the stream is raised to the caller.
        """
        result = TurnResult()

        def consume(on_change) -> None:
            for event in events:
                if isinstance(event, TextDelta):
                    result.text += event.text
                    on_change()
                elif isinstance(event, ReasoningDelta):
                    result.reasoning += event.text
                    on_change()
                elif isinstance(event, ToolCallEvent):
                    result.tool_calls.append(event.call)
                elif isinstance(event, Usage):
                    result.usage = event
                elif isinstance(event, Done):
                    result.finish_reason = event.finish_reason
                    result.assistant_message = event.assistant_message

        try:
            if not self.console.is_terminal:
                consume(lambda: None)
            else:
                last_update = 0.0
                with Live(
                    self._event_view(result, title),
                    console=self.console,
                    transient=True,
                    vertical_overflow="ellipsis",
                    refresh_per_second=8,
                ) as live:
                    def refresh() -> None:
                        nonlocal last_update
                        now = time.monotonic()
                        if now - last_update > self.update_interval:
                            live.update(self._event_view(result, title))
                            last_update = now

                    consume(refresh)
        except KeyboardInterrupt:
            result.interrupted = True
            close = getattr(events, "close", None)
            if close is not None:
                close()

        if result.interrupted:
            if result.text:
                self.console.print(
                    Panel(Markdown(result.text), title="LLMCLI · cancelled", border_style="dim")
                )
            self.print_info("Generation cancelled.")
        elif result.text or (self.show_reasoning and result.reasoning):
            self.console.print(self._event_view(result, title))
        elif show_empty:
            self.console.print(
                Panel(Text("No response received.", style="dim"), title=title, border_style=self.colors["border"])
            )
        return result

    def print_capabilities(self, capabilities: Capabilities, context_size: int) -> None:
        table = Table(show_header=True, header_style=f"bold {self.colors['accent']}", pad_edge=False)
        table.add_column("Capability", no_wrap=True)
        table.add_column("Value")
        table.add_column("Source", style="dim")
        for label, value, source in capabilities.rows():
            table.add_row(label, escape(value), source)
        self.console.print(table)
        length = capabilities.context_length
        if isinstance(length, int) and length != context_size:
            self.print_info(
                f"The server reports a context length of {length}; llmcli uses context_size = "
                f"{context_size} from your configuration."
            )

    def print_usage(self, usage: Usage | None) -> None:
        if usage is None:
            self.console.print("[dim]Last turn:[/dim] none yet")
            return

        def show(value: Any, suffix: str = "") -> str:
            return "?" if value is None else f"{value}{suffix}"

        parts = [
            f"first token {show(usage.ttft_s, ' s')}",
            f"prompt {show(usage.prompt_tokens)} tokens",
            f"generated {show(usage.completion_tokens)} tokens",
            f"{show(usage.gen_tokens_per_s)} tokens/s",
        ]
        if usage.cached_tokens is not None:
            parts.append(f"cached {usage.cached_tokens} tokens")
        self.console.print(f"[dim]Last turn:[/dim] {' · '.join(parts)}")
