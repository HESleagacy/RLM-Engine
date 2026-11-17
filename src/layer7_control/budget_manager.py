"""Token / cost budget tracking."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from shared.model_errors import BudgetExceededError


@dataclass(frozen=True, eq=False)
class Reservation:
    amount: int


@dataclass
class BudgetManager:
    """Soft budget for tokens (or abstract cost units)."""

    limit: int
    used: int = 0
    reserved: int = 0
    uncertain: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _pending: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.limit, self.used, self.reserved, self.uncertain)) or min(self.limit, self.used, self.reserved, self.uncertain) < 0:
            raise ValueError("budget limit and used amount must be non-negative")
        if self.uncertain > self.used:
            raise ValueError("Uncertain usage cannot exceed charged usage")
        if self.reserved:
            raise ValueError("Outstanding reservations cannot be reconstructed from a counter")

    def can_spend(self, n: int) -> bool:
        with self._lock:
            return n >= 0 and self.used + self.reserved + n <= self.limit

    def spend(self, n: int) -> None:
        if n < 0:
            raise ValueError("spend amount must be non-negative")
        with self._lock:
            if self.used + self.reserved + n > self.limit:
                raise BudgetExceededError("budget exceeded")
            self.used += n

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limit - self.used - self.reserved)

    def reserve(self, n: int) -> Reservation:
        if type(n) is not int or n < 0:
            raise ValueError("Reservation must be a non-negative integer")
        with self._lock:
            if self.used + self.reserved + n > self.limit:
                raise BudgetExceededError("Insufficient budget for input plus reserved output")
            reservation = Reservation(n)
            self._pending[reservation] = n
            self.reserved += n
            return reservation

    def settle(self, reservation: Reservation, actual: int | None) -> None:
        if actual is not None and (type(actual) is not int or actual < 0):
            raise ValueError("Actual usage must be a non-negative integer or unknown")
        with self._lock:
            if not isinstance(reservation, Reservation) or reservation not in self._pending:
                raise ValueError("Unknown reservation")
            reserved = self._pending.pop(reservation)
            self.reserved -= reserved
            charged = reserved if actual is None else actual
            self.used += charged
            if actual is None:
                self.uncertain += charged
            if self.used > self.limit:
                # Preserve incurred usage even when the provider exceeds our estimate.
                raise BudgetExceededError("Provider-reported usage exceeded the remaining budget")

    def record_external(self, used: int, uncertain: int = 0) -> None:
        """Reconcile worker usage without losing an overrun or unknown charges."""
        if min(used, uncertain) < 0 or uncertain > used:
            raise ValueError("Invalid external usage")
        with self._lock:
            self.used += used
            self.uncertain += uncertain
