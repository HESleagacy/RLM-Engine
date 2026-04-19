"""Finalization protocol — FINAL() locks output (no regeneration after compute)."""

from __future__ import annotations

from dataclasses import dataclass

from shared.constants import FINAL_SENTINEL


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    text: str
    locked: bool = True


def FINAL(value: str) -> FinalAnswer:
    """Mark the definitive output; callers must treat this as immutable."""
    _ = FINAL_SENTINEL  # reserved for tracing / logging hooks
    return FinalAnswer(text=str(value), locked=True)
