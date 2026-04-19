"""Filtering — regex / keywords."""

from __future__ import annotations

import re

from layer1_input.context_repr import MountedContext


def by_keyword(ctx: MountedContext, *keywords: str) -> list[str]:
    lines = ctx.text.splitlines()
    kws = [k.lower() for k in keywords]
    return [ln for ln in lines if any(k in ln.lower() for k in kws)]


def by_regex(ctx: MountedContext, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [ln for ln in ctx.text.splitlines() if rx.search(ln)]
