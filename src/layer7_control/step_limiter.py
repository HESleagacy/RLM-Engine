"""Hard cap on reasoning / execution steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepLimiter:
    max_steps: int
    current: int = 0

    def allow(self) -> bool:
        return self.current < self.max_steps

    def tick(self) -> None:
        if not self.allow():
            raise RuntimeError("step limit exceeded")
        self.current += 1
