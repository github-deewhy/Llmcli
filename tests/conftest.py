"""Live smoke tests are opt-in: ``pytest -m live --live-profile NAME`` (spec 14, section 3).

Without ``--live-profile`` every test marked ``live`` is skipped, so the default
run and CI never touch the network.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live-profile",
        action="append",
        default=[],
        metavar="NAME",
        help="Run live tests against this profile from ~/.llmcli/config.toml (repeatable).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live-profile"):
        return
    skip = pytest.mark.skip(reason="live test: pass --live-profile NAME to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if "live_profile" in metafunc.fixturenames:
        names = metafunc.config.getoption("--live-profile") or ["(none)"]
        metafunc.parametrize("live_profile", names)
