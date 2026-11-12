"""Tests for streamed terminal output."""

from io import StringIO

from rich.console import Console

from llmcli.ui import StreamingUI


def test_non_terminal_stream_renders_one_complete_response():
    output = StringIO()
    ui = StreamingUI(Console(file=output, force_terminal=False))

    response = ui.render_stream(iter(["A useful ", "**answer**."]))

    rendered = output.getvalue()
    assert response == "A useful **answer**."
    assert "LLMCLI · response" in rendered
    assert rendered.count("LLMCLI · response") == 1
    assert "A useful" in rendered
    assert "answer" in rendered
