"""Manages conversation history and context window trimming."""

from typing import Any

from llmcli.budget import TokenEstimator, TrimResult, messages_chars, reply_reserve


class Conversation:
    """Chat history plus the token budget it must fit in (spec 16, part C).

    ``max_tokens`` is the model's context window (``context_size``). ``reply_tokens``
    is the configured reply length (``max_tokens`` in config.toml); part of it is kept
    free for the reply (see ``budget.reply_reserve``). Messages are not trimmed when
    added: callers call ``fit()`` right before each request, when the fixed part of the
    prompt (system prompt, instructions, tool schemas) is known.
    """

    def __init__(self, system_prompt: str, max_tokens: int, reply_tokens: int | None = None):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.reply_tokens = reply_tokens
        self.messages: list[dict[str, Any]] = []
        self.estimator = TokenEstimator()
        # ids of user messages that continue a turn instead of starting one: text-mode
        # tool results and repair notes. Trimming never cuts a turn at them.
        self._continuations: set[int] = set()
        # How many turns were dropped to fit the budget (spec 15, 2.6).
        self.trim_count = 0

    def add(self, role: str, content: str, continues_turn: bool = False) -> None:
        """Add a message. ``continues_turn=True`` marks a user message that is part of
        the current turn (a text-mode tool result or a repair note), not a new question."""
        self.add_message({"role": role, "content": content}, continues_turn)

    def add_message(self, message: dict[str, Any], continues_turn: bool = False) -> None:
        """Append a provider message, including structured tool calls/results."""
        self._continuations.discard(id(message))
        if continues_turn:
            self._continuations.add(id(message))
        self.messages.append(message)

    def _starts_turn(self, message: dict[str, Any]) -> bool:
        return message.get("role") == "user" and id(message) not in self._continuations

    def prompt_budget(self) -> int:
        """Tokens available for the whole prompt: the window minus the reply reserve."""
        return self.max_tokens - reply_reserve(self.max_tokens, self.reply_tokens)

    def fit(self, fixed_chars: int) -> TrimResult:
        """Drop the oldest whole turns until the estimated prompt fits the budget.

        ``fixed_chars`` is everything sent besides ``messages``: the system message
        as sent and the tool schemas. A turn starts at a user message not marked
        ``continues_turn``; the latest turn is never dropped, so tool calls stay with
        their results and a task prompt is never dropped in the middle of its turn.
        """
        budget = self.prompt_budget()
        fixed = self.estimator.tokens(fixed_chars)
        total = fixed + self.estimator.tokens(messages_chars(self.messages))
        dropped = 0
        freed = 0
        while total > budget:
            next_turn = next(
                (index for index, message in enumerate(self.messages[1:], start=1)
                 if self._starts_turn(message)),
                None,
            )
            if next_turn is None:
                break
            removed = self.estimator.tokens(messages_chars(self.messages[:next_turn]))
            self.messages = self.messages[next_turn:]
            dropped += 1
            freed += removed
            total = fixed + self.estimator.tokens(messages_chars(self.messages))
        self.trim_count += dropped
        return TrimResult(
            turns_dropped=dropped,
            tokens_freed=freed,
            estimated_tokens=total,
            budget=budget,
            over_budget=total > budget,
        )

    def fit_for_chat(self) -> TrimResult:
        """``fit()`` for a plain chat request: the only fixed part is the system prompt."""
        return self.fit(len(self.system_prompt or ""))

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
        continuations: set[int] = set()
        for message in self.messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user" or (role == "assistant" and content.strip()):
                copy = {"role": role, "content": content}
                if id(message) in self._continuations:
                    continuations.add(id(copy))
                kept.append(copy)
        self.messages = kept
        self._continuations = continuations
        return before - len(kept)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self._continuations = set()
