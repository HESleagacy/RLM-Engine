"""Depth-limited sub-call orchestration."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import TYPE_CHECKING

from layer4_recursion.result_integrator import merge_strings
from layer4_recursion.subtask_constructor import from_prompt
from layer7_control.recursion_guard import RecursionGuard
from shared.model_call import call_model, RetryPolicy, estimate_input_tokens
from shared.model_errors import SubcallLimitError

if TYPE_CHECKING:
    from layer7_control.budget_manager import BudgetManager
    from layer7_control.token_tracker import TokenTracker
    from shared.types import LLMCallable


class RecursionManager:
    def __init__(
        self,
        guard: RecursionGuard,
        llm: "LLMCallable",
        budget: "BudgetManager | None" = None,
        token_tracker: "TokenTracker | None" = None,
        max_subcalls_per_step: int = 4,
        retry_policy=None,
        output_tokens=1024,
        token_counter=estimate_input_tokens,
    ) -> None:
        self.guard = guard
        self.llm = llm
        self.budget = budget
        self.token_tracker = token_tracker
        if max_subcalls_per_step < 0:
            raise ValueError("max_subcalls_per_step must be non-negative")
        self.max_subcalls_per_step = max_subcalls_per_step
        self._calls = 0
        self._calls_lock = threading.Lock()
        self.retry_policy = retry_policy or RetryPolicy()
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError("Subcall output-token limit must be positive")
        self.output_tokens, self.token_counter = output_tokens, token_counter
        self.deadline = None

    def begin_step(self):
        with self._calls_lock:
            self._calls = 0

    def run_subtask(self, prompt: str) -> str:
        with self._calls_lock:
            if self._calls >= self.max_subcalls_per_step:
                raise SubcallLimitError("subcall limit exceeded")
            self._calls += 1
        self.guard.enter()
        try:
            task = from_prompt(prompt, depth=self.guard.depth)
            text, tokens = call_model(
                self.llm, task.prompt, budget=self.budget, tracker=self.token_tracker,
                max_tokens=self.output_tokens, deadline=self.deadline, policy=self.retry_policy,
                token_counter=self.token_counter,
            )
            return text
        finally:
            self.guard.leave()

    def run_many(self, prompts: list[str]) -> str:
        self.begin_step()
        out: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.run_subtask, p) for p in prompts]
            for future in concurrent.futures.as_completed(futures):
                out.append(future.result())
        return merge_strings(out)
