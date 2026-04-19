"""S-NIAH — constant complexity needle-in-a-haystack style tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SNIAHExample:
    haystack: str
    needle: str
    expected_span: str | None = None


def trivial_example() -> SNIAHExample:
    return SNIAHExample(haystack="x" * 100 + "ANSWER" + "y" * 100, needle="ANSWER")
