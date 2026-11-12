"""Text tool-call protocol and argument checks (spec 14, section 5, case 10), plus the think splitter."""

import pytest

from llmcli.coding_tools import tool_schemas
from llmcli.providers.think_split import ThinkSplitter
from llmcli.tool_protocol import check_arguments, find_text_call, text_mode_instructions

TOOLS = tool_schemas()


def test_fenced_json_block_becomes_a_tool_call():
    text = 'I will read it.\n```json\n{"name": "read_file", "arguments": {"path": "app.py"}}\n```'
    found = find_text_call(text, TOOLS)
    assert found.error is None
    assert found.call.name == "read_file" and found.call.arguments == {"path": "app.py"}


def test_tool_call_tags_and_bare_object_are_recognised():
    tagged = '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>'
    bare = '{"name": "search_files", "arguments": {"query": "TODO"}}'
    assert find_text_call(tagged, TOOLS).call.name == "list_files"
    assert find_text_call(bare, TOOLS).call.arguments == {"query": "TODO"}


def test_unknown_tool_gives_an_error_for_the_model():
    found = find_text_call('```json\n{"name": "delete_all", "arguments": {}}\n```', TOOLS)
    assert found.call is None and "Unknown tool 'delete_all'" in found.error


def test_bad_arguments_give_a_specific_error():
    found = find_text_call('```json\n{"name": "read_file", "arguments": {"start_line": 2}}\n```', TOOLS)
    assert found.error == 'read_file: "path" missing; expected string.'


@pytest.mark.parametrize(
    "text",
    [
        "Plain answer with no JSON.",
        '```json\n{"name": "my-package", "version": "1.0"}\n```',  # ordinary JSON, no "arguments"
        '```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```\n'
        '```json\n{"name": "read_file", "arguments": {"path": "b"}}\n```',  # two calls: not parsed
        "```python\nprint('hi')\n```",
    ],
)
def test_normal_answers_are_not_tool_calls(text):
    found = find_text_call(text, TOOLS)
    assert found.call is None and found.error is None


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"path": 3}, '"path" must be string, got int.'),
        ({"path": "a", "start_line": True}, '"start_line" must be integer, got boolean.'),
        ({"path": "a", "extra": 1}, '"extra" is not an argument'),
        ({"path": "a", "start_line": 2, "tail": False}, None),
    ],
)
def test_check_arguments(arguments, message):
    schema = next(t for t in TOOLS if t["function"]["name"] == "read_file")["function"]["parameters"]
    result = check_arguments(schema, arguments)
    assert (result is None) if message is None else (message in result)


def test_text_mode_instructions_list_every_tool():
    text = text_mode_instructions(TOOLS)
    for tool in TOOLS:
        assert tool["function"]["name"] in text
    assert "path: string" in text and "start_line?: integer" in text


def test_think_splitter_handles_tags_split_inside():
    splitter = ThinkSplitter()
    out = []
    for chunk in ["Hi <", "think", ">plan</", "think> done <th"]:
        out += splitter.feed(chunk)
    out += splitter.flush()
    joined = {"text": "", "reasoning": ""}
    for kind, piece in out:
        joined[kind] += piece
    assert joined == {"text": "Hi  done <th", "reasoning": "plan"}
