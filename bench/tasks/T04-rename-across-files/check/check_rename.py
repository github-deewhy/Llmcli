from pathlib import Path

import pricing
from cart import Cart
from invoice import invoice_line

ITEMS = [("pen", 2, 150), ("pad", 1, 399)]


def test_new_name_works():
    assert pricing.order_total(ITEMS) == 699


def test_old_name_gone_everywhere():
    for name in ("pricing.py", "cart.py", "invoice.py"):
        assert "calc_total" not in Path(name).read_text(encoding="utf-8"), name


def test_callers_still_work():
    cart = Cart()
    cart.add("pen", 2, 150)
    assert cart.total() == 300
    assert invoice_line(ITEMS) == "Total: 6.99"
