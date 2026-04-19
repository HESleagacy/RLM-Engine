"""BrowseComp — multi-hop reasoning benchmark placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BrowseCompTask:
    question: str
    gold: str


def sample_task() -> BrowseCompTask:
    return BrowseCompTask(question="What links A to B?", gold="C")
