"""Text helpers."""

import re


def normalize_spaces(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim the ends."""
    return re.sub(r"\s+", " ", text).strip()
