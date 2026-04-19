"""Restrict unsafe behavior in generated code."""

from __future__ import annotations

import builtins
from typing import Any


def safe_builtins() -> dict[str, Any]:
    """Minimal builtins for deterministic sandboxed execution."""
    b = builtins.__dict__.copy()
    for name in ("open", "input", "compile", "eval", "exec", "__import__"):
        b.pop(name, None)
    return b
