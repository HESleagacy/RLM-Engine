"""Context Representation — immutable mounted prompt P."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MountedContext:
    """Single string P mounted in the external environment; do not mutate."""

    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text))
