"""Abstract base class for all LLM providers."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCall]
    assistant_message: dict[str, Any]


def parse_tool_calls(calls: list[dict[str, Any]]) -> list[ToolCall]:
    parsed = []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("Provider returned a malformed tool call.")
        function = call.get("function", {})
        if not isinstance(function, dict):
            raise ValueError("Provider returned a malformed tool function.")
        name = function.get("name")
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid arguments for tool {name!r}: {error}") from error
        else:
            arguments = raw_arguments
        if not isinstance(name, str) or not name:
            raise ValueError("Provider returned a tool call without a function name.")
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {name!r} must be a JSON object.")
        parsed.append(ToolCall(str(call.get("id") or uuid4().hex), name, arguments))
    return parsed


class BaseProvider(ABC):
    model: str

    @abstractmethod
    def check_health(self) -> tuple[bool, str]:
        """
        Check if the provider is reachable and the model is available.
        Returns (is_healthy: bool, message: str)
        """
        pass

    @abstractmethod
    def chat(self, messages: list[dict], stream: bool = True) -> Iterator[str]:
        """
        Send messages to the provider and yield response chunks.
        """
        pass

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return one non-streamed response, including any structured tool calls."""
        pass

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        """Format a tool result in the OpenAI-compatible message shape."""
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": content,
        }

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        List available models from the provider.
        """
        pass
