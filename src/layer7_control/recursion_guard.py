"""Prevent recursive explosion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecursionGuard:
    max_depth: int
    depth: int = 0

    def enter(self) -> None:
        if self.depth >= self.max_depth:
            raise RuntimeError("recursion depth exceeded")
        self.depth += 1

    def leave(self) -> None:
        self.depth = max(0, self.depth - 1)
