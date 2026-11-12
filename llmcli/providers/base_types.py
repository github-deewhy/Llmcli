"""Tool-call data type and parsing shared by providers, events and the agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse OpenAI- or Ollama-shaped tool calls. Raises ValueError on a malformed call."""
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
            if not raw_arguments.strip():
                arguments: Any = {}
            else:
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
