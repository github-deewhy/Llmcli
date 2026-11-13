import pricing


class Cart:
    def __init__(self) -> None:
        self.items: list[tuple[str, int, int]] = []

    def add(self, name: str, quantity: int, price: int) -> None:
        self.items.append((name, quantity, price))

    def total(self) -> int:
        return pricing.calc_total(self.items)
