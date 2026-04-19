"""OOLONG-Pairs — quadratic scaling benchmark placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OolongPairsConfig:
    pair_count: int


def default_config() -> OolongPairsConfig:
    return OolongPairsConfig(pair_count=16)
