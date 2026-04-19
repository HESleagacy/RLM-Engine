"""Planning — decide what to read / ignore / stop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PlanAction(Enum):
    READ_CHUNK = auto()
    STOP = auto()
    GENERATE_CODE = auto()


@dataclass
class PlanStep:
    action: PlanAction
    detail: str = ""


class Planner:
    """Minimal planner: stub policy until learned planning lands."""

    def next_step(self, step_index: int, *, has_more: bool) -> PlanStep:
        if not has_more:
            return PlanStep(PlanAction.STOP, "done")
        if step_index == 0:
            return PlanStep(PlanAction.READ_CHUNK, "initial")
        return PlanStep(PlanAction.GENERATE_CODE, "continue")
