"""Depth-limited sub-call orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from layer4_recursion.result_integrator import merge_strings
from layer4_recursion.sub_llm_invoker import invoke
from layer4_recursion.subtask_constructor import Subtask, from_prompt
from layer7_control.recursion_guard import RecursionGuard

if TYPE_CHECKING:
    from shared.types import LLMCallable


class RecursionManager:
    def __init__(self, guard: RecursionGuard, llm: "LLMCallable") -> None:
        self.guard = guard
        self.llm = llm

    def run_subtask(self, prompt: str) -> str:
        self.guard.enter()
        try:
            task = from_prompt(prompt, depth=self.guard.depth)
            return invoke(task, self.llm)
        finally:
            self.guard.leave()

    def run_many(self, prompts: list[str]) -> str:
        out: list[str] = []
        for p in prompts:
            out.append(self.run_subtask(p))
        return merge_strings(out)
