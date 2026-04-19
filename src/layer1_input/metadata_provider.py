"""Metadata Provider — length and structure hints for mounted context."""

from __future__ import annotations

from layer1_input.context_repr import MountedContext
from shared.types import StructureHints
from shared.utils import count_paragraphs, has_code_fence


def describe(ctx: MountedContext) -> StructureHints:
    t = ctx.text
    line_count = 0 if not t else t.count("\n") + 1
    return StructureHints(
        line_count=line_count,
        paragraph_count=count_paragraphs(t),
        has_code_fence=has_code_fence(t),
    )


def byte_length(ctx: MountedContext) -> int:
    return len(ctx.text.encode("utf-8"))
