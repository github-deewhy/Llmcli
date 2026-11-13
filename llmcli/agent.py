"""Approval-driven loop for interactive coding tasks, driven by streamed events (spec 14, 2.6, 2.8)."""

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llmcli.coding_tools import ProjectCodingTools
from llmcli.history import Conversation
from llmcli.providers.base import BaseProvider, ToolCall
from llmcli.providers.events import ProviderError, Usage
from llmcli.tool_protocol import (
    check_arguments,
    find_text_call,
    text_mode_instructions,
    tool_result_text,
)
from llmcli.ui import StreamingUI

MAX_TOOL_ROUNDS = 8
MAX_TOOL_CALLS_PER_ROUND = 4
# In "auto" mode, switch to text mode after this many replies in a row that
# wrote a tool call as text instead of using the native format.
TEXT_CALLS_BEFORE_SWITCH = 2
REPAIRS_PER_TURN = 1
# Longest tool error message kept in ``RunOutcome.tool_error_messages``.
MAX_TOOL_ERROR_CHARS = 200
AGENT_INSTRUCTIONS = (
    "For coding tasks, inspect the relevant project files with tools before making "
    "claims. Use tools for actual file changes and tests; never claim an edit or test "
    "succeeded unless the tool result confirms it. File edits and test execution "
    "require the user's explicit approval."
)
# Added with stop_guard = true (spec 16, B4).
STOP_INSTRUCTION = "When the task is done, give your final answer in plain text without calling a tool."
REPEAT_NOTE = (
    "Not run again: this is the same call with the same arguments as your previous call, "
    "and its result is above. If the task is done, answer in plain text without a tool call."
)
EDIT_TOOL_NAMES = frozenset({"replace_in_file", "edit_lines", "create_file"})


@dataclass
class RunOutcome:
    """How one ``run()`` ended, for the benchmark harness (spec 15, section 2.6).

    ``stop_reason`` is one of: ``answered`` (final text after using tools),
    ``no_tools`` (final text without any tool use), ``round_limit``,
    ``provider_error``, ``interrupted`` (Ctrl-C during a reply) or
    ``cancelled`` (Ctrl-C during a tool).

    ``invalid_calls`` counts calls rejected before running (unknown tool or bad
    arguments). ``tool_errors`` counts calls that ran and returned an error, for
    example a ``replace_in_file`` whose ``old_text`` was not found (spec 16, A3);
    ``tool_error_messages`` holds their messages, one per error. ``repeat_skips``
    counts calls not run because they repeated the previous call (``stop_guard``).
    """

    stop_reason: str = "answered"
    rounds: int = 0
    tool_calls: int = 0
    invalid_calls: int = 0
    tool_errors: int = 0
    tool_error_messages: list[str] = field(default_factory=list)
    repeat_skips: int = 0
    repairs: int = 0
    mode_switched: bool = False
    error: str | None = None
    error_type: str | None = None
    final_text: str = ""
    usages: list[Usage] = field(default_factory=list)


class CodingAgent:
    def __init__(
        self,
        provider: BaseProvider,
        conversation: Conversation,
        tools: ProjectCodingTools,
        ui: StreamingUI,
        confirm: Callable[[str], bool],
        tool_mode: str = "auto",
        stop_guard: bool = False,
    ):
        self.provider = provider
        self.conversation = conversation
        self.tools = tools
        self.ui = ui
        self.confirm = confirm
        self.stop_guard = stop_guard
        self.set_tool_mode(tool_mode)
        self._outcome = RunOutcome()
        self._previous_call: tuple[str, str] | None = None

    def apply_settings(self, edit_matching: str, edit_tools: str, stop_guard: bool) -> None:
        """Agent settings from the active profile (spec 16, part B)."""
        self.tools.edit_matching = edit_matching
        self.tools.edit_tools = edit_tools
        self.stop_guard = stop_guard

    def set_tool_mode(self, tool_mode: str) -> None:
        """``auto`` and ``native`` start with native calls; ``text`` uses the text protocol."""
        self.tool_mode = tool_mode
        self.mode = "text" if tool_mode == "text" else "native"
        self.text_call_streak = 0

    def set_provider(self, provider: BaseProvider, tool_mode: str) -> None:
        self.provider = provider
        self.set_tool_mode(tool_mode)

    def run(self, prompt: str) -> RunOutcome:
        outcome = RunOutcome()
        self._outcome = outcome
        self.conversation.add("user", prompt)
        schemas = self.tools.schemas()
        self._previous_call = None
        used_tools = False
        repairs_left = REPAIRS_PER_TURN
        for _round_number in range(1, MAX_TOOL_ROUNDS + 1):
            text_mode = self.mode == "text"
            outcome.rounds += 1
            try:
                result = self.ui.render_events(
                    self.provider.stream(
                        self._messages_for_agent(text_mode, schemas),
                        None if text_mode else schemas,
                    ),
                    show_empty=False,
                )
            except ProviderError as error:
                self.ui.print_error(f"Agent request failed: {error}")
                outcome.stop_reason = "provider_error"
                outcome.error = str(error)
                outcome.error_type = type(error).__name__
                return outcome
            if result.usage is not None:
                outcome.usages.append(result.usage)

            if result.interrupted:
                if not used_tools and self.conversation.messages[-1:] == [
                    {"role": "user", "content": prompt}
                ]:
                    self.conversation.messages.pop()
                outcome.stop_reason = "interrupted"
                return outcome

            assistant_message = dict(
                result.assistant_message or {"role": "assistant", "content": result.text}
            )
            assistant_message.setdefault("role", "assistant")
            self.conversation.add_message(assistant_message)

            calls = list(result.tool_calls)
            via_text = False
            if calls:
                self.text_call_streak = 0
            elif self.tool_mode != "native":
                found = find_text_call(result.text, schemas)
                if found.call is not None:
                    calls, via_text = [found.call], True
                    self._note_text_call()
                elif found.error is not None:
                    outcome.invalid_calls += 1
                    if repairs_left > 0:
                        repairs_left -= 1
                        outcome.repairs += 1
                        self.ui.print_info(f"Tool call rejected: {found.error}")
                        self.conversation.add(
                            "user", f"Tool call error: {found.error} Reply with a corrected tool call."
                        )
                        continue

            if not calls:
                outcome.final_text = result.text
                if not result.text:
                    self.ui.print_info("The model returned no response.")
                if not used_tools:
                    self.ui.print_info(
                        "No tools were invoked; no files were changed or tests run. "
                        "The selected model may not support tool calling."
                    )
                outcome.stop_reason = "answered" if used_tools else "no_tools"
                return outcome

            used_tools = True
            if not self._run_calls(calls, via_text, schemas):
                outcome.stop_reason = "cancelled"
                return outcome

        self.ui.print_error(
            f"Agent paused after {MAX_TOOL_ROUNDS} tool rounds. "
            "Review the work and send a follow-up to continue."
        )
        outcome.stop_reason = "round_limit"
        return outcome

    def _note_text_call(self) -> None:
        if self.mode == "text":
            return
        self.text_call_streak += 1
        if self.tool_mode == "auto" and self.text_call_streak >= TEXT_CALLS_BEFORE_SWITCH:
            self.mode = "text"
            self._outcome.mode_switched = True
            self.ui.print_info(
                "The model writes tool calls as text; switched to text tool calls for this session "
                '(set tool_mode = "native" in config.toml to prevent this).'
            )

    def _run_calls(self, calls: list[ToolCall], via_text: bool, schemas: list[dict[str, Any]]) -> bool:
        """Execute up to the per-round limit. Returns False if the user cancelled."""
        parameters = {s["function"]["name"]: s["function"].get("parameters", {}) for s in schemas}
        for index, call in enumerate(itertools.islice(calls, MAX_TOOL_CALLS_PER_ROUND)):
            result_recorded = False
            try:
                self.ui.print_info(f"Tool request: [bold]{call.name}[/bold]")
                problem = (
                    check_arguments(parameters[call.name], call.arguments)
                    if call.name in parameters
                    else None
                )
                self._outcome.tool_calls += 1
                if problem or call.name not in parameters:
                    self._outcome.invalid_calls += 1
                signature = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
                repeated = self.stop_guard and signature == self._previous_call
                self._previous_call = signature
                if problem:
                    result = json.dumps({"error": f"{call.name}: {problem}"})
                elif repeated:
                    self._outcome.repeat_skips += 1
                    result = json.dumps({"status": "skipped", "note": REPEAT_NOTE})
                else:
                    result = self.tools.execute(call.name, call.arguments, self.confirm)
                    self._note_tool_error(call.name, result)
                self._record(call, result, via_text)
                result_recorded = True
                self._show_tool_result(call.name, result)
            except KeyboardInterrupt:
                if not result_recorded:
                    self._record(call, '{"error":"Tool execution interrupted by user."}', via_text)
                for pending_call in calls[index + 1:]:
                    self._record(
                        pending_call, '{"error":"Tool execution skipped after interruption."}', via_text
                    )
                self.ui.print_info("Tool execution cancelled.")
                return False
        for call in calls[MAX_TOOL_CALLS_PER_ROUND:]:
            self._record(call, '{"error":"Tool-call limit reached for this response."}', via_text)
        return True

    def _note_tool_error(self, name: str, result: str) -> None:
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict) and "error" in data:
            self._outcome.tool_errors += 1
            message = " ".join(f"{name}: {data['error']}".split())
            self._outcome.tool_error_messages.append(message[:MAX_TOOL_ERROR_CHARS])

    def _record(self, call: ToolCall, result: str, via_text: bool) -> None:
        if via_text:
            self.conversation.add("user", tool_result_text(call.name, result))
        else:
            self.conversation.add_message(self.provider.tool_result_message(call, result))

    def _show_tool_result(self, name: str, result: str) -> None:
        data = json.loads(result)
        status = data.get("status")
        if name in EDIT_TOOL_NAMES:
            if status in {"updated", "created"}:
                self.ui.print_success(f"{status.title()}: [bold]{data['path']}[/bold]")
            elif status == "cancelled":
                self.ui.print_info(f"Edit not applied: [bold]{data['path']}[/bold]")
            elif "error" in data:
                self.ui.print_error(data["error"])
        elif name == "run_tests":
            self.ui.print_test_result(
                data.get("command", "test command"),
                status or "failed",
                data.get("output") or data.get("error", ""),
            )

    def _messages_for_agent(self, text_mode: bool, schemas: list[dict[str, Any]]) -> list[dict]:
        instructions = AGENT_INSTRUCTIONS
        if self.stop_guard:
            instructions = f"{instructions} {STOP_INSTRUCTION}"
        if text_mode:
            instructions = f"{instructions}\n\n{text_mode_instructions(schemas)}"
        messages = self.conversation.get_messages_for_api()
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": f"{messages[0].get('content', '')}\n\n{instructions}",
            }
        else:
            messages.insert(0, {"role": "system", "content": instructions})
        return messages
