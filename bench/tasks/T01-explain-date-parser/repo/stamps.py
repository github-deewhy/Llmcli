"""Helpers for timestamps found in log lines."""

from datetime import date


def read_stamp(text: str) -> date:
    """Turn 'YYYY-MM-DD' into a date."""
    year, month, day = (int(part) for part in text.strip().split("-"))
    return date(year, month, day)
