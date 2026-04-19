"""Persistent variable storage for the execution environment."""

from __future__ import annotations

from typing import Any


class StateStore:
    """Mutable mapping of names to values (survives across exec rounds if reused)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)
