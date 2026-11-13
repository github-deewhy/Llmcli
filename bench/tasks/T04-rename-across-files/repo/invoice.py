import pricing


def invoice_line(items: list[tuple[str, int, int]]) -> str:
    cents = pricing.calc_total(items)
    return f"Total: {cents // 100}.{cents % 100:02d}"
