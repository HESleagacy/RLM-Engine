"""Hard cap on reasoning / execution steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepLimiter:
    max_steps: int
    current: int = 0

    def __post_init__(self) -> None:
        if self.max_steps < 0 or self.current < 0:
            raise ValueError("step limits must be non-negative")

    def allow(self) -> bool:
        return self.current < self.max_steps

    def tick(self) -> None:
        if not self.allow():
            raise RuntimeError("step limit exceeded")
        self.current += 1
