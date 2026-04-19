"""Safe utilities exposed to generated code."""

from __future__ import annotations

from typing import Any, Callable


class ToolInterface:
    """Registry of callable tools (read-only helpers, math, etc.)."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def as_namespace(self) -> dict[str, Any]:
        return dict(self._tools)
