"""Aggregate view of execution health."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from layer7_control.budget_manager import BudgetManager
from layer7_control.recursion_guard import RecursionGuard
from layer7_control.step_limiter import StepLimiter
from layer7_control.token_tracker import TokenTracker


@dataclass
class ExecutionMonitor:
    steps: StepLimiter
    budget: BudgetManager
    recursion: RecursionGuard
    tokens: TokenTracker = field(default_factory=TokenTracker)

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": {"current": self.steps.current, "max": self.steps.max_steps},
            "budget": {"used": self.budget.used, "limit": self.budget.limit},
            "recursion_depth": self.recursion.depth,
            "token_events": len(self.tokens.counts),
        }
