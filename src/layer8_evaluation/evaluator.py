"""Thin evaluation runner over benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from layer8_evaluation.metrics.accuracy import exact_match
from layer8_evaluation.metrics.cost import total_cost


@dataclass(frozen=True, slots=True)
class EvalResult:
    correct: bool
    cost: tuple[int, int]


def evaluate_one(
    gold: str,
    predict: Callable[[], str],
    *,
    tokens: int = 0,
    steps: int = 0,
) -> EvalResult:
    pred = predict()
    ok = exact_match(pred, gold)
    c = total_cost(tokens, steps)
    return EvalResult(correct=ok, cost=(c.tokens, c.steps))
