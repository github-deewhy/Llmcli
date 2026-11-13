from ranges import inclusive_range


def test_inclusive_range_includes_end():
    assert inclusive_range(1, 3) == [1, 2, 3]
