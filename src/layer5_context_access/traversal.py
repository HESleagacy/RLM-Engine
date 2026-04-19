"""Tree / line-graph style navigation over structured text."""

from __future__ import annotations

from dataclasses import dataclass

from layer1_input.context_repr import MountedContext


@dataclass
class LineNode:
    index: int
    text: str
    children: list["LineNode"]


def lines_as_tree(ctx: MountedContext) -> LineNode:
    """Flat list represented as a degenerate tree (future: real AST)."""
    root = LineNode(index=-1, text="", children=[])
    for i, line in enumerate(ctx.text.splitlines()):
        root.children.append(LineNode(index=i, text=line, children=[]))
    return root
