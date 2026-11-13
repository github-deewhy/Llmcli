"""Token estimates for context budgeting (spec 16, part C).

Token counts are estimated from character counts. The estimate starts at
4 characters per token and is calibrated from the prompt token count the
server reports for each request. The factor lives only in memory.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_CHARS_PER_TOKEN = 4.0
MIN_CHARS_PER_TOKEN = 2.0
MAX_CHARS_PER_TOKEN = 6.0
# The reply reserve never takes more than this share of the context window,
# so a large max_tokens cannot leave no room for the conversation.
MAX_REPLY_SHARE = 0.25


def message_chars(message: dict[str, Any]) -> int:
    """Characters of one chat message that count towards the prompt."""
    chars = len(str(message.get("content") or ""))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        chars += len(json.dumps(tool_calls, ensure_ascii=False, default=str))
    return chars


def messages_chars(messages: Iterable[dict[str, Any]]) -> int:
    return sum(message_chars(message) for message in messages)


def schemas_chars(schemas: list[dict[str, Any]] | None) -> int:
    """Characters of the tool schemas sent with a request (0 when none are sent)."""
    if not schemas:
        return 0
    return len(json.dumps(schemas, ensure_ascii=False, default=str))


def reply_reserve(context_size: int, max_tokens: int | None) -> int:
    """Tokens kept free for the reply: ``max_tokens``, at most a quarter of the window."""
    if not max_tokens or max_tokens <= 0 or context_size <= 0:
        return 0
    return min(max_tokens, int(context_size * MAX_REPLY_SHARE))


class TokenEstimator:
    """Characters-to-tokens estimate, calibrated from server-reported prompt tokens."""

    def __init__(self) -> None:
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        self.calibrated = False

    def tokens(self, chars: int) -> int:
        if chars <= 0:
            return 0
        return int(chars / self.chars_per_token)

    def calibrate(self, sent_chars: int, prompt_tokens: int | None) -> None:
        """Use the latest request's ratio, limited to 2-6 characters per token.

        Ignored when the server reported no (or zero) prompt tokens or nothing was sent.
        """
        if not prompt_tokens or prompt_tokens <= 0 or sent_chars <= 0:
            return
        ratio = sent_chars / prompt_tokens
        self.chars_per_token = min(MAX_CHARS_PER_TOKEN, max(MIN_CHARS_PER_TOKEN, ratio))
        self.calibrated = True

    def reset(self) -> None:
        """Back to the default, for example after switching to another model."""
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        self.calibrated = False


@dataclass(frozen=True)
class TrimResult:
    """What fitting the history into the budget did.

    ``turns_dropped``: whole earlier turns removed; ``tokens_freed``: their estimated
    size; ``estimated_tokens``: estimated prompt size afterwards (fixed part plus
    messages); ``budget``: tokens available for the prompt; ``over_budget``: the prompt
    is still larger than the budget because nothing more could be dropped.
    """

    turns_dropped: int = 0
    tokens_freed: int = 0
    estimated_tokens: int = 0
    budget: int = 0
    over_budget: bool = False


def trim_notice(result: TrimResult, context_size: int) -> str | None:
    """One line telling the user that history was trimmed (``05.5``: never trim silently)."""
    if result.turns_dropped <= 0:
        return None
    turns = "turn" if result.turns_dropped == 1 else "turns"
    return (
        f"History trimmed: dropped {result.turns_dropped} oldest {turns} "
        f"(~{result.tokens_freed:,} tokens) to fit context_size {context_size}."
    )


def over_budget_warning(result: TrimResult, context_size: int) -> str | None:
    """One line warning that the request is still larger than the budget after trimming."""
    if not result.over_budget:
        return None
    reserved = context_size - result.budget
    return (
        f"Context nearly full: this request is ~{result.estimated_tokens:,} tokens, over the "
        f"budget of {result.budget:,} (context_size {context_size} minus {reserved:,} kept for "
        "the reply), and there is no older turn left to drop. The server may refuse it or cut "
        "the reply short; /clear starts a new conversation."
    )
