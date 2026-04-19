"""Control flow — loop, branch, terminate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class FlowState(Enum):
    RUNNING = auto()
    TERMINATE = auto()


@dataclass
class ControlFlow:
    max_iterations: int = 10
    iteration: int = 0

    def should_continue(self) -> bool:
        return self.iteration < self.max_iterations

    def advance(self) -> None:
        self.iteration += 1

    def terminate(self) -> FlowState:
        return FlowState.TERMINATE
