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

    def can_spend(self, n: int) -> bool:
        with self._lock:
            return self.used + n <= self.limit

    def spend(self, n: int) -> None:
        with self._lock:
            if self.used + n > self.limit:
                raise RuntimeError("budget exceeded")
            self.used += n

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limit - self.used)
