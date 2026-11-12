"""Approval-driven loop for interactive coding tasks, driven by streamed events (spec 14, 2.6, 2.8)."""

import itertools
import json
from collections.abc import Callable
from typing import Any

from llmcli.coding_tools import ProjectCodingTools, tool_schemas
from llmcli.history import Conversation
from llmcli.providers.base import BaseProvider, ToolCall
from llmcli.providers.events import ProviderError
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
AGENT_INSTRUCTIONS = (
    "For coding tasks, inspect the relevant project files with tools before making "
    "claims. Use tools for actual file changes and tests; never claim an edit or test "
    "succeeded unless the tool result confirms it. File edits and test execution "
    "require the user's explicit approval."
)


class CodingAgent:
    def __init__(
        self,
        provider: BaseProvider,
        conversation: Conversation,
        tools: ProjectCodingTools,
        ui: StreamingUI,
        confirm: Callable[[str], bool],
        tool_mode: str = "auto",
    ):
        self.provider = provider
        self.conversation = conversation
        self.tools = tools
        self.ui = ui
        self.confirm = confirm
        self.set_tool_mode(tool_mode)

    def set_tool_mode(self, tool_mode: str) -> None:
        """``auto`` and ``native`` start with native calls; ``text`` uses the text protocol."""
        self.tool_mode = tool_mode
        self.mode = "text" if tool_mode == "text" else "native"
        self.text_call_streak = 0

    def set_provider(self, provider: BaseProvider, tool_mode: str) -> None:
        self.provider = provider
        self.set_tool_mode(tool_mode)

    def run(self, prompt: str) -> None:
        self.conversation.add("user", prompt)
        schemas = tool_schemas()
        used_tools = False
        repairs_left = REPAIRS_PER_TURN
        for _round_number in range(1, MAX_TOOL_ROUNDS + 1):
            text_mode = self.mode == "text"
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
                return

            if result.interrupted:
                if not used_tools and self.conversation.messages[-1:] == [
                    {"role": "user", "content": prompt}
                ]:
                    self.conversation.messages.pop()
                return

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
                elif found.error is not None and repairs_left > 0:
                    repairs_left -= 1
                    self.ui.print_info(f"Tool call rejected: {found.error}")
                    self.conversation.add(
                        "user", f"Tool call error: {found.error} Reply with a corrected tool call."
                    )
                    continue

            if not calls:
                if not result.text:
                    self.ui.print_info("The model returned no response.")
                if not used_tools:
                    self.ui.print_info(
                        "No tools were invoked; no files were changed or tests run. "
                        "The selected model may not support tool calling."
                    )
                return

            used_tools = True
            if not self._run_calls(calls, via_text, schemas):
                return

        self.ui.print_error(
            f"Agent paused after {MAX_TOOL_ROUNDS} tool rounds. "
            "Review the work and send a follow-up to continue."
        )

    def _note_text_call(self) -> None:
        if self.mode == "text":
            return
        self.text_call_streak += 1
        if self.tool_mode == "auto" and self.text_call_streak >= TEXT_CALLS_BEFORE_SWITCH:
            self.mode = "text"
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
                if problem:
                    result = json.dumps({"error": f"{call.name}: {problem}"})
                else:
                    result = self.tools.execute(call.name, call.arguments, self.confirm)
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

    def _record(self, call: ToolCall, result: str, via_text: bool) -> None:
        if via_text:
            self.conversation.add("user", tool_result_text(call.name, result))
        else:
            self.conversation.add_message(self.provider.tool_result_message(call, result))

    def _show_tool_result(self, name: str, result: str) -> None:
        data = json.loads(result)
        status = data.get("status")
        if name in {"replace_in_file", "create_file"}:
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
