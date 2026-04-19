"""OOLONG — linear scaling benchmark placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OolongConfig:
    length_tokens: int


def default_config() -> OolongConfig:
    return OolongConfig(length_tokens=1024)
