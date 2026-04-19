"""Combine intermediate pieces into one candidate answer."""

from __future__ import annotations

from typing import Any, Sequence


def join_text(parts: Sequence[Any], sep: str = "\n") -> str:
    return sep.join(str(p) for p in parts)
