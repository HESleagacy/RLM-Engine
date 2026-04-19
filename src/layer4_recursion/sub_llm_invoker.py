"""Invoke sub-LLM with optional handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from layer4_recursion.subtask_constructor import Subtask

if TYPE_CHECKING:
    from shared.types import LLMCallable


def invoke(task: Subtask, llm: "LLMCallable") -> str:
    return llm(task.prompt)
