from text_utils import normalize_spaces, slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_runs_become_one_hyphen():
    assert slugify("Hello,   World!!") == "hello-world"


def test_trims_hyphens():
    assert slugify("  --Already-Slugged--  ") == "already-slugged"


def test_digits_kept():
    assert slugify("Version 2 Notes") == "version-2-notes"


def test_empty():
    assert slugify("!!!") == ""


def test_existing_function_unchanged():
    assert normalize_spaces("  a   b ") == "a b"
