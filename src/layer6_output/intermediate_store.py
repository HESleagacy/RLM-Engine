"""Intermediate storage before finalization."""

from __future__ import annotations

from typing import Any


class IntermediateStore:
    def __init__(self) -> None:
        self._items: list[Any] = []

    def append(self, item: Any) -> None:
        self._items.append(item)

    def all(self) -> tuple[Any, ...]:
        return tuple(self._items)
