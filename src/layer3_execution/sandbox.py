"""Restrict unsafe behavior in generated code."""

from __future__ import annotations

import builtins
import ast
from typing import Any


def safe_builtins() -> dict[str, Any]:
    """Return an allowlist of harmless builtins for generated code."""
    names = (
        "abs", "all", "any", "bool", "dict", "enumerate", "Exception", "filter",
        "float", "int", "isinstance", "len", "list", "map", "max", "min", "print",
        "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "ValueError", "zip",
    )
    return {name: getattr(builtins, name) for name in names}


class SandboxViolation(ValueError):
    """Raised when generated code contains an unsafe operation."""


def validate_code(source: str) -> None:
    """Reject imports and dunder introspection before generated code executes."""
    try:
        tree = ast.parse(source, filename="<generated>", mode="exec")
    except SyntaxError:
        raise

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation("imports are not available in the execution sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxViolation("private and dunder attributes are not available")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise SandboxViolation("private and dunder names are not available")
