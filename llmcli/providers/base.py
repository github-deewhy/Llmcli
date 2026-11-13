"""Provider interface: one streaming method plus probing (spec 14, section 2.1)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from llmcli.providers.base_types import ToolCall, parse_tool_calls
from llmcli.providers.events import (
    Capabilities,
    Event,
    ReasoningDelta,
    TextDelta,
    TextIsReasoning,
    ToolCallEvent,
    Usage,
)
from llmcli.providers.http import DEFAULT_REQUEST_TIMEOUT_S, make_client
from llmcli.providers.think_split import ThinkSplitter

__all__ = ["BaseProvider", "ToolCall", "parse_tool_calls", "StreamState"]

PROBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Reply to a ping.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    },
}
PROBE_PROMPT = "Call the ping tool with value 1. Do not answer in text."
# Client-side speed is reported only above these, so a reply that arrives in one
# burst does not show an absurd tokens/s (spec 16, A2).
MIN_SPEED_TOKENS = 8
MIN_SPEED_SECONDS = 0.5


class StreamState:
    """Per-request bookkeeping: answer text, inline-reasoning split, timings."""

    def __init__(self, capabilities: Capabilities):
        self.capabilities = capabilities
        self.started = time.monotonic()
        self.first_token_at: float | None = None
        self.text = ""
        self.splitter = ThinkSplitter()
        self.saw_reasoning_field = False

    def _mark_first(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()

    def content(self, chunk: str) -> Iterator[Event]:
        """Answer text, with any inline ``<think>`` part split out."""
        if chunk:
            self._mark_first()
        for kind, piece in self.splitter.feed(chunk):
            yield from self._emit(kind, piece)

    def reasoning(self, chunk: str) -> Iterator[Event]:
        if chunk:
            self._mark_first()
            if not self.saw_reasoning_field:
                self.saw_reasoning_field = True
                self.capabilities.set("reasoning", "field", "observed")
            yield ReasoningDelta(chunk)

    def tool_call(self, call: ToolCall) -> ToolCallEvent:
        self._mark_first()
        self.capabilities.set("native_tools", "yes", "observed")
        return ToolCallEvent(call)

    def finish(self) -> Iterator[Event]:
        for kind, piece in self.splitter.flush():
            yield from self._emit(kind, piece)
        if self.splitter.seen_tag and not self.saw_reasoning_field:
            self.capabilities.set("reasoning", "inline", "observed")

    def _emit(self, kind: str, piece: str) -> Iterator[Event]:
        if kind == "retract":
            # The text emitted so far was reasoning (orphan ``</think>``).
            if self.text:
                self.text = ""
                yield TextIsReasoning()
            return
        if not piece:
            return
        if kind == "reasoning":
            yield ReasoningDelta(piece)
        else:
            self.text += piece
            yield TextDelta(piece)

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round(self.first_token_at - self.started, 3)

    def wall_tokens_per_s(self, completion_tokens: int | None) -> float | None:
        """Generation speed from client timing, when the server gives no duration.

        ``None`` unless at least ``MIN_SPEED_TOKENS`` arrived over at least
        ``MIN_SPEED_SECONDS``; below that the number measures buffering, not generation.
        """
        if completion_tokens is None or completion_tokens < MIN_SPEED_TOKENS or self.first_token_at is None:
            return None
        elapsed = time.monotonic() - self.first_token_at
        if elapsed < MIN_SPEED_SECONDS:
            return None
        return round(completion_tokens / elapsed, 2)


class BaseProvider(ABC):
    """All providers stream typed events; nothing else talks HTTP to a model server."""

    model: str
    endpoint: str

    def __init__(self, request_timeout: int = DEFAULT_REQUEST_TIMEOUT_S):
        self.request_timeout = request_timeout
        self.client = make_client(request_timeout)
        self.capabilities = Capabilities()
        self.last_usage: Usage | None = None

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[Event]:
        """Send one request and yield events. Raises ``ProviderError`` subclasses."""

    @abstractmethod
    def probe(self) -> Capabilities:
        """Read server metadata (no generation). Never raises; unknown stays unknown."""

    @abstractmethod
    def check_health(self) -> tuple[bool, str]:
        """Return (reachable and model available, message)."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Model names the server offers; empty list on failure."""

    def live_probe(self) -> Capabilities:
        """Run one tiny generation with one dummy tool (``/probe`` only, P4)."""
        saw_call = False
        saw_usage = False
        for event in self.stream([{"role": "user", "content": PROBE_PROMPT}], [PROBE_TOOL]):
            if isinstance(event, ToolCallEvent) and event.call.name == "ping":
                saw_call = True
            elif isinstance(event, Usage):
                saw_usage = True
        self.capabilities.set("native_tools", "yes" if saw_call else "no", "live probe")
        self.capabilities.set("streams_tool_calls", "yes" if saw_call else "no", "live probe")
        self.capabilities.set("usage_reported", "yes" if saw_usage else "no", "live probe")
        return self.capabilities

    def reset_capabilities(self) -> None:
        self.capabilities = Capabilities()
        self.last_usage = None

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        """Format a tool result in the OpenAI-compatible message shape."""
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def record_usage(self, usage: Usage) -> Usage:
        self.last_usage = usage
        if usage.prompt_tokens is not None or usage.completion_tokens is not None:
            self.capabilities.set("usage_reported", "yes", "observed")
        return usage

    def close(self) -> None:
        self.client.close()
