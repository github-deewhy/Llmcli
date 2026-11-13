from loader import load_settings
from server import describe
from settings import Settings


def test_default_timeout():
    assert Settings().timeout == 30
    assert load_settings({}).timeout == 30


def test_timeout_read_from_dict():
    assert load_settings({"timeout": "45"}).timeout == 45


def test_describe_shows_timeout():
    assert describe(load_settings({"host": "h", "port": 1, "timeout": 5})) == "h:1 (timeout 5s)"


def test_existing_fields_unchanged():
    s = load_settings({"host": "example", "port": "9000"})
    assert (s.host, s.port) == ("example", 9000)
