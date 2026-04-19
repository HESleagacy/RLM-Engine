"""Merge sub-results into parent context."""

from __future__ import annotations

from typing import Sequence


def merge_strings(parts: Sequence[str], sep: str = "\n---\n") -> str:
    return sep.join(p.strip() for p in parts if p.strip())
