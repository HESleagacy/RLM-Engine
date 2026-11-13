"""Segmented reads — chunking."""

from __future__ import annotations

from typing import Iterator

from layer1_input.context_repr import MountedContext


def iter_windows(ctx: MountedContext, size: int, overlap: int = 0) -> Iterator[str]:
    """Yield windows on demand, without collecting the document in RAM."""
    if size <= 0:
        raise ValueError("size must be positive")
    t = ctx.text
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    step = size - overlap
    i = 0
    while i < len(t):
        yield t[i : i + size]
        i += step


def fixed_windows(ctx: MountedContext, size: int, overlap: int = 0) -> list[str]:
    """Compatibility helper; use iter_windows for memory-bounded traversal."""
    return list(iter_windows(ctx, size, overlap))
