"""Resolves @file attachments in user prompts."""

import re
from pathlib import Path

from llmcli.security import CREDENTIAL_DIRECTORIES, IGNORED_DIRECTORIES, filename_block_reason

MAX_ATTACHMENT_BYTES = 512_000

# '@' must start a token (not follow a word character, '.', or another '@'),
# so e-mail addresses like bob@example.com are left alone.
ATTACHMENT_PATTERN = re.compile(r"(?<![\w.@])@([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)")


def _block_reason(path: Path) -> str | None:
    reason = filename_block_reason(path)
    if reason:
        return reason
    resolved = path.resolve()
    if any(part.lower() in CREDENTIAL_DIRECTORIES for part in resolved.parts):
        return "Access to credential directories is blocked."
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return None  # outside the project: only name/credential rules apply
    if any(part.lower() in IGNORED_DIRECTORIES for part in relative.parts):
        return "Access to generated or version-control directories is blocked."
    return None


def resolve_attachments(text: str) -> str:
    """
    Replace @path/to/file with a fenced block of that file's contents.

    Text that merely looks like an attachment (missing files, e-mail addresses)
    is left unchanged. Blocked or oversized files are replaced by a short notice
    instead of their contents.
    """

    def replacer(match: re.Match) -> str:
        filepath = match.group(1)
        path = Path(filepath)
        if not path.is_file():
            return match.group(0)

        reason = _block_reason(path)
        if reason:
            return f"{match.group(0)} *[Not attached: {reason}]*"
        try:
            if path.stat().st_size > MAX_ATTACHMENT_BYTES:
                return f"{match.group(0)} *[Not attached: file exceeds {MAX_ATTACHMENT_BYTES} bytes]*"
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            return f"\n\n*[Error reading {filepath}: {error}]*\n\n"
        ext = path.suffix.lstrip(".") or "text"
        return f"\n\n--- Content of {filepath} ---\n```{ext}\n{content}\n```\n--- End of {filepath} ---\n\n"

    return ATTACHMENT_PATTERN.sub(replacer, text)
