"""Prevent recursive explosion with thread-local depth accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class RecursionGuard:
    max_depth: int
    _local: threading.local = field(default_factory=threading.local, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    @property
    def depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def enter(self) -> None:
        if self.depth >= self.max_depth:
            raise RuntimeError("recursion depth exceeded")
        self._local.depth = self.depth + 1

    def leave(self) -> None:
        self._local.depth = max(0, self.depth - 1)
