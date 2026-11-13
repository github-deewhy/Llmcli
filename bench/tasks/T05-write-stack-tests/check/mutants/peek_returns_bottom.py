"""A minimal stack."""


class Stack:
    def __init__(self) -> None:
        self._items: list = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raises IndexError if empty."""
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it. Raises IndexError if empty."""
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[0]

    def is_empty(self) -> bool:
        return not self._items
