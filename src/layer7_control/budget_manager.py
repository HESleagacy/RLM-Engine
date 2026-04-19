"""Token / cost budget tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetManager:
    """Soft budget for tokens (or abstract cost units)."""

    limit: int
    used: int = 0

    def can_spend(self, n: int) -> bool:
        return self.used + n <= self.limit

    def spend(self, n: int) -> None:
        if not self.can_spend(n):
            raise RuntimeError("budget exceeded")
        self.used += n

    def remaining(self) -> int:
        return max(0, self.limit - self.used)
