"""Small reads — probing the mounted context."""

from __future__ import annotations

from layer1_input.context_repr import MountedContext


def peek_head(ctx: MountedContext, n: int = 200) -> str:
    return ctx.text[:n]


def peek_tail(ctx: MountedContext, n: int = 200) -> str:
    return ctx.text[-n:] if n else ""
