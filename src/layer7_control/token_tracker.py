"""Optional alias-style tracker; pairs with BudgetManager."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading


@dataclass
class TokenTracker:
    counts: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, n: int) -> None:
        if n < 0:
            raise ValueError("token count must be non-negative")
        with self._lock:
            self.counts.append(n)

    def record_many(self, values: list[int]) -> None:
        for value in values:
            self.record(value)

    def since(self, index: int) -> list[int]:
        with self._lock:
            return list(self.counts[index:])

    def total(self) -> int:
        return sum(self.counts)
