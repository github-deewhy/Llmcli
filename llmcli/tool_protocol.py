"""Text tool-call protocol and argument checks (fallback ladder rung 3, spec 14, section 2.6).

Small models often write a tool call as text instead of using the server's native
tool-call format. This module recognises exactly one such call in a reply:

    ```json
    {"name": "read_file", "arguments": {"path": "app.py"}}
    ```

or the same object inside ``<tool_call>…</tool_call>``, or a reply that is only
that object. Anything else is treated as a normal answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from llmcli.providers.base_types import ToolCall

FENCED = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)
TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass(frozen=True)
class TextCall:
    """Result of scanning a reply. ``call`` set: a valid call. ``error`` set: tell the model."""

    call: ToolCall | None = None
    error: str | None = None


def _candidates(text: str) -> list[Any]:
    blocks = [m.group(1) for m in TAGGED.finditer(text)] + [m.group(1) for m in FENCED.finditer(text)]
    stripped = text.strip()
    if not blocks and stripped.startswith("{") and stripped.endswith("}"):
        blocks = [stripped]
    found = []
    for block in blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "name" in value and "arguments" in value:
            found.append(value)
    return found


def find_text_call(text: str, tools: list[dict[str, Any]]) -> TextCall:
    """Return a valid call, an error for the model, or neither (a normal answer)."""
    candidates = _candidates(text)
    if len(candidates) != 1:
        return TextCall()
    value = candidates[0]
    name = value["name"]
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    if not isinstance(name, str) or name not in schemas:
        return TextCall(error=f"Unknown tool {name!r}. Available tools: {', '.join(sorted(schemas))}.")
    arguments = value["arguments"]
    if not isinstance(arguments, dict):
        return TextCall(error=f'"arguments" for {name} must be a JSON object.')
    problem = check_arguments(schemas[name], arguments)
    if problem:
        return TextCall(error=f"{name}: {problem}")
    return TextCall(call=ToolCall(f"text_{uuid4().hex[:8]}", name, arguments))


def check_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """A short, specific message for the first problem found, or None if the arguments fit."""
    properties = schema.get("properties", {}) or {}
    for key in schema.get("required", []) or []:
        if key not in arguments:
            expected = properties.get(key, {}).get("type", "a value")
            return f'"{key}" missing; expected {expected}.'
    for key, value in arguments.items():
        if key not in properties:
            if schema.get("additionalProperties") is False:
                return f'"{key}" is not an argument; allowed: {", ".join(sorted(properties))}.'
            continue
        expected = properties[key].get("type")
        allowed = JSON_TYPES.get(expected or "")
        if allowed is None:
            continue
        if isinstance(value, bool) and expected in ("integer", "number"):
            return f'"{key}" must be {expected}, got boolean.'
        if not isinstance(value, allowed):
            return f'"{key}" must be {expected}, got {type(value).__name__}.'
    return None


def text_mode_instructions(tools: list[dict[str, Any]]) -> str:
    """Short system-prompt text describing the tools for text mode."""
    lines = [
        "To use a tool, reply with only one JSON block and nothing else:",
        '```json\n{"name": "<tool>", "arguments": {...}}\n```',
        "You will get the result in the next message. Tools:",
    ]
    for tool in tools:
        function = tool["function"]
        properties = function.get("parameters", {}).get("properties", {})
        required = set(function.get("parameters", {}).get("required", []))
        params = ", ".join(
            f"{key}{'' if key in required else '?'}: {spec.get('type', 'any')}"
            for key, spec in properties.items()
        )
        lines.append(f"- {function['name']}({params}): {function.get('description', '')}")
    return "\n".join(lines)


def tool_result_text(name: str, result: str) -> str:
    """How a tool result is passed back in text mode (as a user message)."""
    return f"Tool result for {name}:\n{result}"
