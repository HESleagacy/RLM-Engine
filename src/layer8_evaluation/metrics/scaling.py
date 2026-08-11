"""Scaling behavior helpers."""

from __future__ import annotations

from typing import Sequence
import math


def approx_complexity(sizes: Sequence[int], costs: Sequence[float]) -> str:
    """Very rough label for demo; real fits need regression."""
    if len(sizes) < 2 or len(sizes) != len(costs):
        return "unknown"
    if any(size <= 0 or cost <= 0 for size, cost in zip(sizes, costs)):
        return "unknown"
    slopes = [
        math.log(costs[i] / costs[i - 1]) / math.log(sizes[i] / sizes[i - 1])
        for i in range(1, len(sizes))
        if sizes[i] != sizes[i - 1]
    ]
    if not slopes:
        return "unknown"
    exponent = sum(slopes) / len(slopes)
    if exponent < 0.25:
        return "sublinear_or_constant"
    if exponent < 1.5:
        return "linear_or_subquadratic"
    return "superlinear"
