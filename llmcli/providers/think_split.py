"""Split inline ``<think>…</think>`` text out of streamed answer content (spec 14, section 2.2).

Some servers put a thinking model's reasoning inside the answer text. Tags can
arrive split across chunks (``"<thi"`` + ``"nk>"``), so a short tail that could
be the start of a tag is held back until the next chunk decides it.

Some models never write the opening tag: the reply starts with reasoning and
the first tag is ``</think>``. Then everything before it was reasoning, and the
splitter returns a ``("retract", "")`` pair so the caller can move the text it
already emitted to reasoning (llmcli 0.5.1).
"""

from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"


def _partial_tag_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for size in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0


class ThinkSplitter:
    """Feed content chunks; get back ``("text" | "reasoning", piece)`` pairs."""

    def __init__(self) -> None:
        self.in_think = False
        self.pending = ""
        self.seen_tag = False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        data = self.pending + chunk
        self.pending = ""
        while data:
            if not self.in_think and not self.seen_tag:
                close_index = data.find(CLOSE)
                open_index = data.find(OPEN)
                if close_index >= 0 and (open_index < 0 or close_index < open_index):
                    out.append(("retract", ""))
                    if close_index:
                        out.append(("reasoning", data[:close_index]))
                    data = data[close_index + len(CLOSE):]
                    self.seen_tag = True
                    continue
            tag = CLOSE if self.in_think else OPEN
            index = data.find(tag)
            if index >= 0:
                if index:
                    out.append(("reasoning" if self.in_think else "text", data[:index]))
                data = data[index + len(tag):]
                self.in_think = not self.in_think
                self.seen_tag = True
                continue
            keep = _partial_tag_suffix(data, tag)
            if not self.in_think and not self.seen_tag:
                keep = max(keep, _partial_tag_suffix(data, CLOSE))
            emit = data[: len(data) - keep] if keep else data
            if emit:
                out.append(("reasoning" if self.in_think else "text", emit))
            self.pending = data[len(data) - keep:] if keep else ""
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Release anything held back at the end of the stream."""
        if not self.pending:
            return []
        piece, self.pending = self.pending, ""
        return [("reasoning" if self.in_think else "text", piece)]
