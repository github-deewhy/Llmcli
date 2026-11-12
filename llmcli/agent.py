"""Approval-driven loop for interactive coding tasks."""

import itertools
import json
from typing import Callable

import httpx

from llmcli.coding_tools import ProjectCodingTools, tool_schemas
from llmcli.history import Conversation
from llmcli.providers.base import BaseProvider
from llmcli.ui import StreamingUI

MAX_TOOL_ROUNDS = 8
MAX_TOOL_CALLS_PER_ROUND = 4
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
    ):
        self.provider = provider
        self.conversation = conversation
        self.tools = tools
        self.ui = ui
        self.confirm = confirm

    def run(self, prompt: str) -> None:
        self.conversation.add("user", prompt)
        used_tools = False
        for _round_number in range(1, MAX_TOOL_ROUNDS + 1):
            try:
                with self.ui.show_spinner("Thinking…"):
                    response = self.provider.complete(
                        self._messages_for_agent(),
                        tool_schemas(),
                    )
            except KeyboardInterrupt:
                if not used_tools and self.conversation.messages[-1:] == [
                    {"role": "user", "content": prompt}
                ]:
                    self.conversation.messages.pop()
                self.ui.print_info("Generation cancelled.")
                return
            except (httpx.HTTPError, ValueError) as error:
                self.ui.print_error(f"Agent request failed: {error}")
                return

            assistant_message = dict(response.assistant_message)
            assistant_message.setdefault("role", "assistant")
            self.conversation.add_message(assistant_message)

            if not response.tool_calls:
                if response.content:
                    self.ui.render_stream(iter((response.content,)))
                else:
                    self.ui.print_info("The model returned no response.")
                if not used_tools:
                    self.ui.print_info(
                        "No tools were invoked; no files were changed or tests run. "
                        "The selected model may not support tool calling."
                    )
                return

            used_tools = True
            for index, call in enumerate(
                itertools.islice(response.tool_calls, MAX_TOOL_CALLS_PER_ROUND)
            ):
                result_recorded = False
                try:
                    self.ui.print_info(f"Tool request: [bold]{call.name}[/bold]")
                    result = self.tools.execute(call.name, call.arguments, self.confirm)
                    self.conversation.add_message(
                        self.provider.tool_result_message(call, result)
                    )
                    result_recorded = True
                    self._show_tool_result(call.name, result)
                except KeyboardInterrupt:
                    if not result_recorded:
                        self.conversation.add_message(
                            self.provider.tool_result_message(
                                call,
                                '{"error":"Tool execution interrupted by user."}',
                            )
                        )
                    for pending_call in response.tool_calls[index + 1:]:
                        self.conversation.add_message(
                            self.provider.tool_result_message(
                                pending_call,
                                '{"error":"Tool execution skipped after interruption."}',
                            )
                        )
                    self.ui.print_info("Tool execution cancelled.")
                    return
            for call in response.tool_calls[MAX_TOOL_CALLS_PER_ROUND:]:
                result = '{"error":"Tool-call limit reached for this response."}'
                self.conversation.add_message(
                    self.provider.tool_result_message(call, result)
                )

        self.ui.print_error(
            f"Agent paused after {MAX_TOOL_ROUNDS} tool rounds. "
            "Review the work and send a follow-up to continue."
        )

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

    def _messages_for_agent(self) -> list[dict]:
        messages = self.conversation.get_messages_for_api()
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": f"{messages[0].get('content', '')}\n\n{AGENT_INSTRUCTIONS}",
            }
        else:
            messages.insert(0, {"role": "system", "content": AGENT_INSTRUCTIONS})
        return messages
