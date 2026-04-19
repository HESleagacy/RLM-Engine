"""Orchestrates intermediate → aggregate → FINAL."""

from __future__ import annotations

from layer6_output.aggregator import join_text
from layer6_output.finalizer import FINAL, FinalAnswer
from layer6_output.intermediate_store import IntermediateStore


class OutputManager:
    def __init__(self) -> None:
        self.intermediate = IntermediateStore()

    def finalize_joined(self, sep: str = "\n") -> FinalAnswer:
        text = join_text(self.intermediate.all(), sep=sep)
        return FINAL(text)
