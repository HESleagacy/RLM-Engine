"""Build subtasks for sub-LLM calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Subtask:
    prompt: str
    depth: int = 0


def from_prompt(prompt: str, depth: int = 0) -> Subtask:
    return Subtask(prompt=prompt, depth=depth)
