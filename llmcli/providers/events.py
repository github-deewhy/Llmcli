"""Typed streaming events, capability record and provider errors (spec 14, sections 2.1, 2.4, 2.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmcli.providers.base_types import ToolCall


@dataclass(frozen=True)
class TextDelta:
    """A piece of the answer text."""

    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    """A piece of the model's thinking text. Never stored in history."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """One complete tool call (fragments are assembled by the provider)."""

    call: ToolCall


@dataclass(frozen=True)
class Usage:
    """Token counts and timings for one request. Any field may be unknown (None)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    ttft_s: float | None = None
    gen_tokens_per_s: float | None = None


@dataclass(frozen=True)
class TextIsReasoning:
    """The answer text sent so far in this reply was reasoning.

    Some thinking models (for example Qwen3 4B Thinking 2507 on Ollama) start
    reasoning without an opening ``<think>`` tag and end it with ``</think>``.
    On this event, consumers move the text they collected so far to reasoning.
    """


@dataclass(frozen=True)
class Done:
    """End of one reply. ``assistant_message`` is what goes into history (no reasoning)."""

    finish_reason: str | None
    assistant_message: dict[str, Any]


Event = TextDelta | ReasoningDelta | TextIsReasoning | ToolCallEvent | Usage | Done


# --- capabilities ------------------------------------------------------------

UNKNOWN = "unknown"

CAPABILITY_LABELS: dict[str, str] = {
    "context_length": "Context length",
    "native_tools": "Native tool calls",
    "streams_tool_calls": "Streams tool calls",
    "reasoning": "Reasoning",
    "usage_reported": "Usage reported",
    "structured_output": "Structured output",
    "parallel_slots": "Parallel slots",
}


@dataclass
class Capabilities:
    """What the server and model can do, with where each value came from.

    Values are ``"unknown"`` until a probe or an observed reply fills them.
    Sources: ``server`` (metadata), ``observed`` (seen in a reply), ``live probe``.
    """

    context_length: int | str = UNKNOWN
    native_tools: str = UNKNOWN  # yes | no | unknown
    streams_tool_calls: str = UNKNOWN  # yes | no | unknown
    reasoning: str = UNKNOWN  # none | field | inline | unknown
    usage_reported: str = UNKNOWN  # yes | no | unknown
    structured_output: str = UNKNOWN  # not used in M1
    parallel_slots: int | str = UNKNOWN
    sources: dict[str, str] = field(default_factory=dict)

    def set(self, name: str, value: Any, source: str) -> None:
        if name not in CAPABILITY_LABELS:
            raise KeyError(name)
        setattr(self, name, value)
        self.sources[name] = source

    def rows(self) -> list[tuple[str, str, str]]:
        """(label, value, source) for display."""
        return [
            (label, str(getattr(self, name)), self.sources.get(name, ""))
            for name, label in CAPABILITY_LABELS.items()
        ]


# --- errors ------------------------------------------------------------------

class ProviderError(Exception):
    """Base class. ``str(error)`` is a one-line message safe to show the user."""


class ProviderConnectError(ProviderError):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderHTTPError(ProviderError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class ProviderProtocolError(ProviderError):
    pass
