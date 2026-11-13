"""Reasoning that ends with ``</think>`` but never opened one (llmcli 0.5.1).

Found in the 2026-11-13 benchmark smoke run: Ollama's ``qwen3:4b`` (Qwen3 4B
Thinking 2507) reasons in the answer text even with ``think = false`` and only
writes the closing tag. That text was stored in history and re-sent each round.
"""

from io import StringIO
from pathlib import Path

import httpx
from rich.console import Console

from llmcli.agent import CodingAgent
from llmcli.bench import environment
from llmcli.coding_tools import ProjectCodingTools
from llmcli.history import Conversation
from llmcli.providers.events import ReasoningDelta, TextDelta, TextIsReasoning
from llmcli.providers.ollama import OllamaProvider
from llmcli.providers.openai_compat import OpenAICompatProvider
from llmcli.providers.think_split import ThinkSplitter
from llmcli.ui import StreamingUI
from tests.test_providers import make_config, ndjson, sse, with_transport


def feed_all(pieces):
    splitter = ThinkSplitter()
    out = []
    for piece in pieces:
        out += splitter.feed(piece)
    return out + splitter.flush()


def joined(out, kind):
    return "".join(piece for k, piece in out if k == kind)


# --- splitter -------------------------------------------------------------------------

def test_orphan_close_retracts_earlier_text():
    out = feed_all(["Let me check ", "the files.\n</think>\n\nThe answer."])
    assert ("retract", "") in out
    retract_at = out.index(("retract", ""))
    assert joined(out[:retract_at], "text") == "Let me check "
    assert joined(out[retract_at:], "reasoning") == "the files.\n"
    assert joined(out[retract_at:], "text") == "\n\nThe answer."


def test_orphan_close_split_across_chunks():
    out = feed_all(["thinking</th", "ink>Done."])
    retract_at = out.index(("retract", ""))
    assert joined(out[:retract_at], "text") == "thinking"  # retracted; "</th" itself was held back
    assert "</th" not in joined(out, "text") + joined(out, "reasoning")
    assert joined(out[retract_at:], "text") == "Done."


def test_normal_think_block_unchanged():
    out = feed_all(["<think>step</think>Answer"])
    assert ("retract", "") not in out
    assert joined(out, "reasoning") == "step" and joined(out, "text") == "Answer"


def test_close_after_a_complete_block_is_plain_text():
    out = feed_all(["<think>a</think>Use the </think> tag."])
    assert ("retract", "") not in out
    assert joined(out, "text") == "Use the </think> tag."


def test_text_without_tags_unchanged():
    out = feed_all(["Just an ", "answer."])
    assert out and all(kind == "text" for kind, _ in out)
    assert joined(out, "text") == "Just an answer."


# --- providers ------------------------------------------------------------------------

def test_ollama_orphan_close_keeps_reasoning_out_of_history():
    def handler(request):
        return httpx.Response(200, text=ndjson(
            {"message": {"role": "assistant", "content": "Okay, let me list "}},
            {"message": {"role": "assistant", "content": "the files.\n</think>\n\n", "tool_calls": [
                {"function": {"name": "list_files", "arguments": {"path": "."}}}]}},
            {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
        ))

    provider = with_transport(OllamaProvider(make_config("ollama", extra_body={"think": False})), handler)
    events = list(provider.stream([{"role": "user", "content": "hi"}], [{"type": "function"}]))

    assert any(isinstance(e, TextIsReasoning) for e in events)
    done = events[-1]
    assert done.assistant_message["content"] == "\n\n"
    assert "let me list" not in done.assistant_message["content"]
    assert done.assistant_message["tool_calls"][0]["function"]["name"] == "list_files"
    assert provider.capabilities.reasoning == "inline"


def test_openai_orphan_close():
    pieces = ["I should ", "answer.</think>", "Paris."]

    def handler(request):
        return httpx.Response(200, text=sse(*({"choices": [{"delta": {"content": p}}]} for p in pieces)))

    provider = with_transport(OpenAICompatProvider(make_config("openai_compat", "http://h/v1")), handler)
    events = list(provider.stream([{"role": "user", "content": "x"}]))
    assert events[-1].assistant_message["content"] == "Paris."
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    assert reasoning == "answer."  # "I should " was already sent as text, then retracted
    assert [type(e).__name__ for e in events].count("TextIsReasoning") == 1


# --- UI and agent -------------------------------------------------------------------

def make_ui():
    return StreamingUI(Console(file=StringIO(), force_terminal=False, width=120))


def test_ui_moves_retracted_text_to_reasoning():
    events = iter([TextDelta("thinking "), TextIsReasoning(), ReasoningDelta("more"), TextDelta("Answer")])
    result = make_ui().render_events(events)
    assert result.text == "Answer"
    assert result.reasoning == "thinking more"


def test_agent_history_and_final_answer_exclude_orphan_reasoning(tmp_path):
    from llmcli.providers.events import Done, Usage
    from llmcli.providers.fake import ScriptedProvider

    reply = [TextDelta("Let me think. "), TextIsReasoning(), ReasoningDelta("Done thinking."),
             TextDelta("It is greet."), Usage(prompt_tokens=5, completion_tokens=5),
             Done("stop", {"role": "assistant", "content": "It is greet."})]
    ui = make_ui()
    conversation = Conversation("ctx", max_tokens=8192)
    (tmp_path / "app.py").write_text("def greet(): pass\n")
    agent = CodingAgent(ScriptedProvider([reply]), conversation, ProjectCodingTools(tmp_path, ui), ui,
                        lambda _m: True)
    outcome = agent.run("what?")
    assert outcome.final_text == "It is greet."
    assert "think" not in str(conversation.messages).lower()


# --- benchmark commit without the git binary -----------------------------------------

def make_git_dir(root: Path, packed: bool = False) -> str:
    sha = "0123456789abcdef0123456789abcdef01234567"
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    if packed:
        (git / "packed-refs").write_text(f"# pack-refs with: peeled\n{sha} refs/heads/main\n")
    else:
        (git / "refs" / "heads" / "main").write_text(sha + "\n")
    return sha


def test_commit_read_from_files_when_git_is_missing(tmp_path, monkeypatch):
    sha = make_git_dir(tmp_path)
    monkeypatch.setattr(environment, "_git", lambda *args: None)
    assert environment.git_info(tmp_path) == {"commit": sha, "dirty": None}


def test_commit_read_from_packed_refs(tmp_path, monkeypatch):
    sha = make_git_dir(tmp_path, packed=True)
    monkeypatch.setattr(environment, "_git", lambda *args: None)
    assert environment.git_info(tmp_path)["commit"] == sha


def test_no_git_folder_gives_none(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "_git", lambda *args: None)
    assert environment.git_info(tmp_path) == {"commit": None, "dirty": None}
