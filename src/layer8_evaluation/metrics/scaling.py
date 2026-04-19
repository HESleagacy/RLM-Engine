"""Scaling behavior helpers."""

from __future__ import annotations

from typing import Sequence


def approx_complexity(sizes: Sequence[int], costs: Sequence[float]) -> str:
    """Very rough label for demo; real fits need regression."""
    if len(sizes) < 2 or len(sizes) != len(costs):
        return "unknown"
    r = costs[-1] / max(costs[0], 1e-9)
    if r < 2:
        return "sublinear_or_constant"
    if r < sizes[-1] / max(sizes[0], 1):
        return "linear_or_subquadratic"
    return "superlinear"
