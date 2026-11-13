"""Builds a daily summary from log lines."""

from collections import Counter

import stamps


def count_per_day(lines: list[str]) -> Counter:
    days = Counter()
    for line in lines:
        days[stamps.read_stamp(line.split(" ", 1)[0])] += 1
    return days
