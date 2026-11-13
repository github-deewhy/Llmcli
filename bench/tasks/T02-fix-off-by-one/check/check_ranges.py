from ranges import evens, inclusive_range


def test_includes_both_ends():
    assert inclusive_range(1, 3) == [1, 2, 3]


def test_single_value():
    assert inclusive_range(5, 5) == [5]


def test_empty_when_reversed():
    assert inclusive_range(3, 1) == []


def test_evens_includes_end():
    assert evens(1, 6) == [2, 4, 6]
