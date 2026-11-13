"""Metadata Provider — length and structure hints for mounted context."""

from __future__ import annotations

from layer1_input.context_repr import MountedContext
from layer1_input.file_context import FileBackedText
from shared.types import StructureHints
from shared.utils import count_paragraphs, has_code_fence


def describe(ctx: MountedContext) -> StructureHints:
    t = ctx.text
    if isinstance(t, FileBackedText):
        newline_count = paragraph_count = 0
        has_content = False
        newline_seen = False
        code_fence = False
        tail = ""
        for start in range(0, len(t), t.block_chars):
            block = t[start:start + t.block_chars]
            newline_count += block.count("\n")
            code_fence = code_fence or "```" in tail + block
            tail = (tail + block)[-2:]
            # Equivalent to splitting on \n\s*\n, without materializing text.
            for char in block:
                if char == "\n":
                    if newline_seen and has_content:
                        paragraph_count += 1
                        has_content = False
                    newline_seen = True
                elif not char.isspace():
                    has_content = True
                    newline_seen = False
        paragraph_count += has_content
        line_count = newline_count + 1 if len(t) else 0
        return StructureHints(line_count, paragraph_count, code_fence)
    line_count = 0 if not t else t.count("\n") + 1
    return StructureHints(
        line_count=line_count,
        paragraph_count=count_paragraphs(t),
        has_code_fence=has_code_fence(t),
    )


def byte_length(ctx: MountedContext) -> int:
    if isinstance(ctx.text, FileBackedText):
        return ctx.text.byte_length
    return len(ctx.text.encode("utf-8"))
