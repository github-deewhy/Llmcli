"""Small range helpers."""


def inclusive_range(start: int, end: int) -> list[int]:
    """All integers from start to end, both included."""
    return list(range(start, end))


def evens(start: int, end: int) -> list[int]:
    """Even integers from start to end, both included."""
    return [n for n in inclusive_range(start, end) if n % 2 == 0]
