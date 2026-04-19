"""Cross-layer type aliases and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

JSONish = Any


@dataclass(frozen=True, slots=True)
class StructureHints:
    """Lightweight hints about prompt structure (Layer 1 metadata)."""

    line_count: int = 0
    paragraph_count: int = 0
    has_code_fence: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Outcome of running generated code in the execution environment."""

    ok: bool
    value: Any = None
    error: str | None = None
    locals_snapshot: Mapping[str, Any] = field(default_factory=dict)


class LLMCallable(Protocol):
    """Pluggable LLM for controller / sub-calls (inject in integration tests)."""

    def __call__(self, prompt: str, *, max_tokens: int | None = None) -> str: ...


SubtaskHandler = Callable[[str, Mapping[str, Any]], str]
