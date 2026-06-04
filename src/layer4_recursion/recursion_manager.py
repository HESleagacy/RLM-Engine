"""Depth-limited sub-call orchestration."""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from layer4_recursion.result_integrator import merge_strings
from layer4_recursion.sub_llm_invoker import invoke
from layer4_recursion.subtask_constructor import Subtask, from_prompt
from layer7_control.recursion_guard import RecursionGuard

if TYPE_CHECKING:
    from layer7_control.budget_manager import BudgetManager
    from shared.types import LLMCallable


class RecursionManager:
    def __init__(
        self,
        guard: RecursionGuard,
        llm: "LLMCallable",
        budget: "BudgetManager | None" = None,
    ) -> None:
        self.guard = guard
        self.llm = llm
        self.budget = budget

    def run_subtask(self, prompt: str) -> str:
        self.guard.enter()
        try:
            task = from_prompt(prompt, depth=self.guard.depth)
            text, tokens = invoke(task, self.llm)
            if self.budget:
                self.budget.spend(tokens)
            return text
        finally:
            self.guard.leave()

    def run_many(self, prompts: list[str]) -> str:
        out: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.run_subtask, p) for p in prompts]
            for future in concurrent.futures.as_completed(futures):
                out.append(future.result())
        return merge_strings(out)
