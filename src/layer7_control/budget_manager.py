"""Token / cost budget tracking."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class BudgetManager:
    """Soft budget for tokens (or abstract cost units)."""

    limit: int
    used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.limit < 0 or self.used < 0:
            raise ValueError("budget limit and used amount must be non-negative")

    def can_spend(self, n: int) -> bool:
        with self._lock:
            return self.used + n <= self.limit

    def spend(self, n: int) -> None:
        if n < 0:
            raise ValueError("spend amount must be non-negative")
        with self._lock:
            if self.used + n > self.limit:
                raise RuntimeError("budget exceeded")
            self.used += n

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limit - self.used)
