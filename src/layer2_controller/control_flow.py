"""Explicit lifecycle; terminal states cannot resume without a new run."""

from dataclasses import dataclass
from enum import Enum


class FlowState(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"
    TERMINATE = "cancelled"  # Compatibility alias.


@dataclass
class ControlFlow:
    max_iterations: int = 20
    iteration: int = 0
    state: FlowState = FlowState.RUNNING
    stop_reason: str | None = None

    def __post_init__(self):
        if type(self.max_iterations) is not int or type(self.iteration) is not int or self.max_iterations < 0 or self.iteration < 0:
            raise ValueError("Iteration limits must be non-negative")

    def should_continue(self) -> bool:
        return self.state == FlowState.RUNNING and self.iteration < self.max_iterations

    def advance(self) -> None:
        if not self.should_continue():
            raise RuntimeError("Cannot advance a terminated or exhausted flow")
        self.iteration += 1

    def reset(self) -> None:
        self.iteration = 0
        self.state = FlowState.RUNNING
        self.stop_reason = None

    def finish(self, state: FlowState, reason: str) -> None:
        if state == FlowState.RUNNING:
            raise ValueError("finish requires a terminal state")
        if self.state != FlowState.RUNNING:
            return
        self.state, self.stop_reason = state, reason

    def terminate(self) -> FlowState:
        self.finish(FlowState.CANCELLED, "cancelled")
        return self.state
