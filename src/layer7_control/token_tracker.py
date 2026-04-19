"""Optional alias-style tracker; pairs with BudgetManager."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenTracker:
    counts: list[int] = field(default_factory=list)

    def record(self, n: int) -> None:
        self.counts.append(n)

    def total(self) -> int:
        return sum(self.counts)
