"""Live smoke tests against a real server (spec 14, section 3). Only Ollama is tested by us.

Run one server at a time, for example:

    pytest -m live --live-profile ollama-qwen3 -v

The profile is read from ~/.llmcli/config.toml. Small models on CPU can take
minutes per test; failures here describe the model as much as llmcli.
"""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from llmcli.agent import CodingAgent
from llmcli.coding_tools import ProjectCodingTools
from llmcli.config import load_config
from llmcli.history import Conversation
from llmcli.providers import create_provider
from llmcli.providers.events import Done, TextDelta
from llmcli.ui import StreamingUI

pytestmark = pytest.mark.live

SYSTEM = "You are a coding assistant. Use the tools to inspect and change files."


@pytest.fixture
def live(live_profile):
    # The fixture project lives in a temp dir, but HOME must stay real so the
    # user's ~/.llmcli/config.toml is read.
    try:
        config = load_config({}, profile=live_profile)
    except ValueError as error:
        pytest.fail(f"{error} Add a [profiles.{live_profile}] table to ~/.llmcli/config.toml.", pytrace=False)
    provider = create_provider(config)
    healthy, message = provider.check_health()
    if not healthy:
        pytest.skip(f"profile {live_profile!r} not reachable: {message}")
    provider.probe()
    yield config, provider
    provider.close()


def make_agent(config, provider, root: Path):
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False, width=120))
    conversation = Conversation(SYSTEM, max_tokens=config.context_size)
    agent = CodingAgent(
        provider, conversation, ProjectCodingTools(root, ui), ui, lambda _m: True, config.tool_mode
    )
    return agent, conversation, output


def tool_results(conversation):
    """Tool results in either form: native tool messages or text-mode user messages."""
    return [
        m for m in conversation.messages
        if m.get("role") == "tool"
        or (m.get("role") == "user" and str(m.get("content", "")).startswith("Tool result for "))
    ]


def test_streams_plain_answer(live):
    _, provider = live
    events = list(provider.stream([{"role": "user", "content": "Count from 1 to 5, separated by spaces."}]))
    assert sum(isinstance(e, TextDelta) for e in events) >= 2
    assert isinstance(events[-1], Done)


def test_reads_a_file_with_a_tool(live, tmp_path):
    config, provider = live
    (tmp_path / "hello.py").write_text("def greet_user(name):\n    return f'Hi {name}'\n")
    agent, conversation, output = make_agent(config, provider, tmp_path)

    agent.run("Read hello.py with the read_file tool and tell me the name of the function it defines.")

    assert tool_results(conversation), f"no tool was used:\n{output.getvalue()}"
    final = conversation.messages[-1]
    assert final["role"] == "assistant" and "greet_user" in str(final.get("content")), output.getvalue()


def test_edits_a_file_with_approval(live, tmp_path):
    config, provider = live
    target = tmp_path / "settings.py"
    target.write_text("MAX_RETRIES = 1\n")
    agent, conversation, output = make_agent(config, provider, tmp_path)

    agent.run(
        "In settings.py, change MAX_RETRIES = 1 to MAX_RETRIES = 3 using the replace_in_file tool."
    )

    assert target.read_text() == "MAX_RETRIES = 3\n", output.getvalue()


def test_probe_fills_server_capabilities(live):
    config, provider = live
    caps = provider.capabilities
    if config.provider == "ollama":
        assert caps.native_tools in ("yes", "no") and isinstance(caps.context_length, int)
    provider.live_probe()
    assert caps.sources.get("native_tools") == "live probe"
