"""Cost metrics — tokens, steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    tokens: int
    steps: int


def total_cost(tokens: int, steps: int) -> CostSnapshot:
    return CostSnapshot(tokens=tokens, steps=steps)
