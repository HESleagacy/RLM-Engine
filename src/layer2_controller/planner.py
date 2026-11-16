"""Planning — decide what to read / ignore / stop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import ast
import hashlib
import json


class PlanAction(Enum):
    READ_CHUNK = auto()
    STOP = auto()
    GENERATE_CODE = auto()


@dataclass
class PlanStep:
    action: PlanAction
    detail: str = ""


class Planner:
    """Host-owned phase guidance and bounded repeated/no-progress detection."""

    def __init__(self, max_repeated_actions=2, max_stagnant_steps=3):
        if max_repeated_actions < 1 or max_stagnant_steps < 1:
            raise ValueError("Progress thresholds must be positive")
        self.max_repeated_actions = max_repeated_actions
        self.max_stagnant_steps = max_stagnant_steps
        self.reset()

    def reset(self):
        self.phase = "inspect"
        self.signatures = {}
        self.last_progress = None
        self.stagnant_steps = 0

    def repeated(self, code: str, state_summary: dict, evidence_count: int) -> bool:
        try:
            normalized = ast.dump(ast.parse(code), include_attributes=False)
        except (SyntaxError, RecursionError):
            normalized = code.strip()
        key = hashlib.sha256(json.dumps(
            [normalized, state_summary, evidence_count], sort_keys=True
        ).encode()).hexdigest()
        self.signatures[key] = self.signatures.get(key, 0) + 1
        return self.signatures[key] > self.max_repeated_actions

    def observe(self, *, ok, state_summary, evidence_count, has_candidate, phase=None):
        progress = json.dumps([state_summary, evidence_count], sort_keys=True)
        self.stagnant_steps = self.stagnant_steps + 1 if progress == self.last_progress else 0
        self.last_progress = progress
        if ok:
            current = phase or self.phase
            self.phase = (
                "synthesize" if has_candidate else
                "retrieve" if current == "inspect" else "analyze"
            )
        return self.stagnant_steps >= self.max_stagnant_steps

    def next_step(self, step_index: int, *, has_more: bool) -> PlanStep:
        if not has_more:
            return PlanStep(PlanAction.STOP, "done")
        if step_index == 0:
            return PlanStep(PlanAction.READ_CHUNK, "initial")
        return PlanStep(PlanAction.GENERATE_CODE, "continue")
