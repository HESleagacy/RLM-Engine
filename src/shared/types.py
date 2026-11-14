"""Cross-layer type aliases and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

JSONish = Any


@dataclass(frozen=True, slots=True)
class PageMarker:
    number: int
    start_char: int


@dataclass(frozen=True, slots=True)
class HeadingHint:
    title: str
    level: int
    start_char: int


@dataclass(frozen=True, slots=True)
class StructureHints:
    """Content-derived statistics; page/heading previews are capped at 32 items."""

    line_count: int = 0
    paragraph_count: int = 0
    has_code_fence: bool = False
    character_count: int = 0
    byte_count: int = 0
    word_count: int = 0
    estimated_tokens: int = 0
    page_count: int = 0
    heading_count: int = 0
    page_markers: tuple[PageMarker, ...] = ()
    headings: tuple[HeadingHint, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Outcome of running generated code in the execution environment."""

    ok: bool
    value: Any = None
    error: str | None = None
    locals_snapshot: Mapping[str, Any] = field(default_factory=dict)
    # Gap 3: stdout captured from print() calls inside exec()
    stdout: str = ""
    error_code: str | None = None


class LLMCallable(Protocol):
    """Single-turn LLM: prompt → (response, tokens). Used by CodeGenerator / RecursionManager."""

    def __call__(self, prompt: str, *, max_tokens: int | None = None) -> tuple[str, int]: ...


class ChatCallable(Protocol):
    """Multi-turn chat LLM: list[dict] messages → (response, tokens). Used in the RLM REPL loop."""

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, int]: ...


SubtaskHandler = Callable[[str, Mapping[str, Any]], str]
