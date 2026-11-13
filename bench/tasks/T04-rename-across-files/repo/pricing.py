"""Prices in cents."""


def calc_total(items: list[tuple[str, int, int]]) -> int:
    """Sum of quantity * unit price for (name, quantity, unit_price) items."""
    return sum(quantity * price for _name, quantity, price in items)
