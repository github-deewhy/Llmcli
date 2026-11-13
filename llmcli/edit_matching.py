"""Help for edits whose ``old_text`` does not match the file exactly (spec 16, B1 and B2).

B1, ``closest_lines``: when ``old_text`` is not found, show the most similar lines
of the file with their numbers, so the model can copy the exact text.

B2, ``tolerant_match``: try once more with trailing whitespace ignored and escaped
quotes (``\\"``, ``\\'``) undone. The edit applies only if exactly one place in
the file matches; the result always says what was ignored. Escaped newlines
(``\\n``) are deliberately not undone: real source code contains them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

MAX_CLOSEST_LINES = 15
MAX_SHOWN_LINE = 200
ESCAPED_QUOTES = (('\\"', '"'), ("\\'", "'"))


def closest_lines(original: str, old_text: str) -> str | None:
    """The file lines most similar to ``old_text``, numbered like ``read_file``; None for an empty file."""
    lines = original.splitlines()
    if not lines:
        return None
    wanted = old_text.strip("\n").splitlines() or [old_text]
    width = min(len(wanted), MAX_CLOSEST_LINES, len(lines))
    target = "\n".join(line.strip() for line in wanted[:width])
    best_start, best_ratio = 0, -1.0
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for start in range(len(lines) - width + 1):
        window = "\n".join(line.strip() for line in lines[start:start + width])
        matcher.set_seq1(window)
        if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_start, best_ratio = start, ratio
    # Show the best window plus context, up to MAX_CLOSEST_LINES lines in total.
    context = (MAX_CLOSEST_LINES - width) // 2
    first = max(0, best_start - context)
    last = min(len(lines), first + MAX_CLOSEST_LINES)
    first = max(0, last - MAX_CLOSEST_LINES)
    shown = []
    for number in range(first, last):
        line = lines[number]
        if len(line) > MAX_SHOWN_LINE:
            line = line[:MAX_SHOWN_LINE] + " …"
        shown.append(f"{number + 1}: {line}")
    return "\n".join(shown)


def not_found_message(original: str, old_text: str) -> str:
    """The error for an ``old_text`` found 0 times, with the closest lines when there are any."""
    message = "Expected old_text exactly once, found it 0 times."
    closest = closest_lines(original, old_text)
    if closest is None:
        return message
    return (
        f"{message} The most similar lines in the file are below (number: text). "
        "Copy old_text exactly from them, without the line numbers.\n"
        f"{closest}"
    )


@dataclass(frozen=True)
class TolerantMatch:
    start: int
    end: int
    new_text: str
    ignored: str  # what was ignored, for the tool result


def _unescape_quotes(text: str) -> str:
    for escaped, plain in ESCAPED_QUOTES:
        text = text.replace(escaped, plain)
    return text


def _pattern(old_text: str) -> re.Pattern[str]:
    """``old_text`` with trailing spaces and tabs ignored at the end of every line."""
    parts = [re.escape(line.rstrip(" \t")) for line in old_text.split("\n")]
    return re.compile(r"[ \t]*\n".join(parts))


def tolerant_match(original: str, old_text: str, new_text: str) -> tuple[TolerantMatch | None, int]:
    """Find ``old_text`` with the B2 rules. Returns (the match if exactly one place matches, places found).

    If the match needed escaped quotes undone, ``new_text`` is unescaped the same
    way, because a model that escapes one argument twice escapes both.
    """
    variants = [(old_text, new_text, "trailing whitespace")]
    unescaped = _unescape_quotes(old_text)
    if unescaped != old_text:
        variants.append((unescaped, _unescape_quotes(new_text), "trailing whitespace and escaped quotes"))
    spans: dict[tuple[int, int], TolerantMatch] = {}
    for old, new, ignored in variants:
        if not old.strip():
            continue
        for found in _pattern(old).finditer(original):
            span = (found.start(), found.end())
            if found.end() > found.start() and span not in spans:
                spans[span] = TolerantMatch(found.start(), found.end(), new, ignored)
    if len(spans) == 1:
        return next(iter(spans.values())), 1
    return None, len(spans)
