"""Segmented reads — chunking."""

from __future__ import annotations

from layer1_input.context_repr import MountedContext


def fixed_windows(ctx: MountedContext, size: int, overlap: int = 0) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    t = ctx.text
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks: list[str] = []
    i = 0
    while i < len(t):
        chunks.append(t[i : i + size])
        i += step
    return chunks
