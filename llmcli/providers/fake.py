"""Scripted provider for tests (and later the M2 harness). Talks to no server."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from llmcli.providers.base import BaseProvider, ToolCall
from llmcli.providers.events import Capabilities, Done, Event, TextDelta, ToolCallEvent, Usage

# Put this object in a script to raise Ctrl-C at that point in the stream.
INTERRUPT = object()


def text_turn(text: str) -> list[Any]:
    """A reply that only contains text, split into two chunks."""
    half = len(text) // 2
    return [
        TextDelta(text[:half]),
        TextDelta(text[half:]),
        Usage(prompt_tokens=10, completion_tokens=5),
        Done("stop", {"role": "assistant", "content": text}),
    ]


def tool_turn(*calls: ToolCall, text: str = "") -> list[Any]:
    """A reply with native tool calls (OpenAI message shape)."""
    import json

    raw = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
        for call in calls
    ]
    events: list[Any] = [TextDelta(text)] if text else []
    events += [ToolCallEvent(call) for call in calls]
    events.append(Done("tool_calls", {"role": "assistant", "content": text, "tool_calls": raw}))
    return events


class ScriptedProvider(BaseProvider):
    """Plays back one scripted list of events per ``stream()`` call and records requests."""

    def __init__(self, turns: Sequence[Sequence[Any]], model: str = "fake-model"):
        super().__init__()
        self.turns = list(turns)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []
        self.model = model
        self.endpoint = "fake://"
        self.closed_streams = 0

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[Event]:
        self.requests.append(([dict(m) for m in messages], tools))
        if not self.turns:
            raise AssertionError("ScriptedProvider ran out of turns")
        script = self.turns.pop(0)
        try:
            for item in script:
                if item is INTERRUPT:
                    raise KeyboardInterrupt
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, Usage):
                    self.record_usage(item)
                yield item
        finally:
            self.closed_streams += 1

    def probe(self) -> Capabilities:
        return self.capabilities

    def check_health(self) -> tuple[bool, str]:
        return True, "fake provider"

    def list_models(self) -> list[str]:
        return [self.model]
