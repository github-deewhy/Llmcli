"""Tests for the interactive tool-calling agent loop."""

import json
from io import StringIO

from rich.console import Console

from llmcli.agent import CodingAgent
from llmcli.coding_tools import ProjectCodingTools
from llmcli.history import Conversation
from llmcli.providers.base import ToolCall
from llmcli.providers.events import Done, TextDelta, ToolCallEvent
from llmcli.providers.fake import INTERRUPT, ScriptedProvider
from llmcli.ui import StreamingUI


def ModelResponse(content, tool_calls, assistant_message):  # noqa: N802 - keeps the old test shape
    """One scripted reply in the event form used since M1."""
    events = [TextDelta(content)] if content else []
    events += [ToolCallEvent(call) for call in tool_calls]
    events.append(Done("stop", assistant_message))
    return events


def FakeProvider(responses):  # noqa: N802
    return ScriptedProvider(responses)


def test_agent_reads_files_then_uses_tool_result_in_followup(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    call = ToolCall("call-1", "read_file", {"path": "app.py"})
    provider = FakeProvider(
        [
            ModelResponse(
                content="",
                tool_calls=[call],
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"app.py"}',
                            },
                        }
                    ],
                },
            ),
            ModelResponse(
                content="The file prints hello.",
                tool_calls=[],
                assistant_message={"role": "assistant", "content": "The file prints hello."},
            ),
        ]
    )
    conversation = Conversation("Coding context", max_tokens=8192)
    agent = CodingAgent(
        provider,
        conversation,
        ProjectCodingTools(root, ui),
        ui,
        lambda _message: False,
    )

    agent.run("Explain app.py")

    assert len(provider.requests) == 2
    assert provider.requests[0][1]
    assert any(message.get("role") == "tool" for message in provider.requests[1][0])
    assert "The file prints hello." in output.getvalue()


def test_agent_reports_when_model_does_not_call_tools(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    provider = FakeProvider(
        [
            ModelResponse(
                content="I changed the file.",
                tool_calls=[],
                assistant_message={"role": "assistant", "content": "I changed the file."},
            )
        ]
    )
    agent = CodingAgent(
        provider,
        Conversation("Coding context", max_tokens=8192),
        ProjectCodingTools(root, ui),
        ui,
        lambda _message: True,
    )

    agent.run("Change app.py")

    assert "No tools were invoked" in output.getvalue()
    assert "I changed the file." in output.getvalue()


def test_agent_cancels_interrupted_model_request_without_losing_prior_history(tmp_path):
    def InterruptingProvider(_responses):  # noqa: N802
        return ScriptedProvider([[INTERRUPT]])

    root = tmp_path / "project"
    root.mkdir()
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    conversation = Conversation("Coding context", max_tokens=8192)
    conversation.add("user", "Earlier question")
    conversation.add("assistant", "Earlier answer")
    previous_messages = list(conversation.messages)
    agent = CodingAgent(
        InterruptingProvider([]),
        conversation,
        ProjectCodingTools(root, ui),
        ui,
        lambda _message: False,
    )

    agent.run("clear")

    assert conversation.messages == previous_messages
    assert "Generation cancelled." in output.getvalue()


def test_agent_records_interrupted_tool_result_before_returning_to_prompt(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    call = ToolCall("call-1", "read_file", {"path": "app.py"})
    provider = FakeProvider(
        [
            ModelResponse(
                content="",
                tool_calls=[call],
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                },
            )
        ]
    )
    tools = ProjectCodingTools(root, ui)

    def interrupt_tool(*_args):
        raise KeyboardInterrupt

    tools.execute = interrupt_tool
    conversation = Conversation("Coding context", max_tokens=8192)
    agent = CodingAgent(provider, conversation, tools, ui, lambda _message: False)

    agent.run("Read app.py")

    assert conversation.messages[-1]["role"] == "tool"
    assert "interrupted by user" in conversation.messages[-1]["content"]
    assert "Tool execution cancelled." in output.getvalue()


def test_agent_can_edit_multiple_files_after_individual_approvals(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "first.py").write_text("value = 1\n", encoding="utf-8")
    (root / "second.py").write_text("value = 1\n", encoding="utf-8")
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))
    calls = [
        ToolCall("call-1", "replace_in_file", {
            "path": "first.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
        }),
        ToolCall("call-2", "replace_in_file", {
            "path": "second.py",
            "old_text": "value = 1",
            "new_text": "value = 3",
        }),
    ]
    provider = FakeProvider(
        [
            ModelResponse(
                content="",
                tool_calls=calls,
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in calls
                    ],
                },
            ),
            ModelResponse(
                content="Both changes are complete.",
                tool_calls=[],
                assistant_message={"role": "assistant", "content": "Both changes are complete."},
            ),
        ]
    )
    confirmations = []
    agent = CodingAgent(
        provider,
        Conversation("Coding context", max_tokens=8192),
        ProjectCodingTools(root, ui),
        ui,
        lambda message: confirmations.append(message) or True,
    )

    agent.run("Change both values.")

    assert (root / "first.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (root / "second.py").read_text(encoding="utf-8") == "value = 3\n"
    assert len(confirmations) == 2
    assert "Updated: first.py" in output.getvalue()
    assert "Updated: second.py" in output.getvalue()
