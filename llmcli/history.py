"""Manages conversation history and context window trimming."""

import json
from typing import Any


class Conversation:
    def __init__(self, system_prompt: str, max_tokens: int):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []

    def add(self, role: str, content: str) -> None:
        """Add a message and automatically trim if necessary."""
        self.add_message({"role": role, "content": content})

    def add_message(self, message: dict[str, Any]) -> None:
        """Append a provider message, including structured tool calls/results."""
        self.messages.append(message)
        self._trim()

    def _trim(self) -> None:
        """
        Trim oldest user/assistant pairs if estimated tokens exceed max_tokens.
        The system prompt is never dropped (it is injected at API time, not stored here).
        """
        # Rough estimation: 1 token ~= 4 characters
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        # Calculate total tokens of current history
        total_tokens = sum(
            estimate_tokens(str(m.get("content") or ""))
            + estimate_tokens(json.dumps(m.get("tool_calls", []), ensure_ascii=False))
            for m in self.messages
        )

        if total_tokens <= self.max_tokens:
            return

        # Drop complete older user turns so tool-call/result message pairs stay intact.
        while total_tokens > self.max_tokens:
            next_turn = next(
                (index for index, message in enumerate(self.messages[1:], start=1)
                 if message.get("role") == "user"),
                None,
            )
            if next_turn is None:
                break
            self.messages = self.messages[next_turn:]
            total_tokens = sum(
                estimate_tokens(str(message.get("content") or ""))
                + estimate_tokens(json.dumps(message.get("tool_calls", []), ensure_ascii=False))
                for message in self.messages
            )

    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """Return messages formatted for the API, with system prompt prepended."""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(self.messages)
        return result

    def keep_text_only(self) -> int:
        """
        Drop tool calls and tool results, keeping plain user/assistant text.

        Used when switching providers, because tool-message formats differ between
        providers. Returns the number of messages removed.
        """
        before = len(self.messages)
        kept = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user" or (role == "assistant" and content.strip()):
                kept.append({"role": role, "content": content})
        self.messages = kept
        return before - len(kept)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
