"""Context Representation — immutable mounted prompt P."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from layer1_input.file_context import FileBackedText


@dataclass(frozen=True, slots=True)
class MountedContext:
    """Single string P mounted in the external environment; do not mutate."""

    text: str | FileBackedText

    def __post_init__(self) -> None:
        if not isinstance(self.text, FileBackedText):
            object.__setattr__(self, "text", str(self.text))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        memory_threshold_bytes: int = 1_048_576,
        block_chars: int = 65_536,
        cache_blocks: int = 8,
    ) -> "MountedContext":
        """Keep small files in RAM; index larger UTF-8 files for lazy reads."""
        if memory_threshold_bytes < 0:
            raise ValueError("memory_threshold_bytes must be non-negative")
        if block_chars <= 0 or cache_blocks < 0:
            raise ValueError("block_chars must be positive and cache_blocks non-negative")
        path = Path(path).resolve(strict=True)
        if path.stat().st_size <= memory_threshold_bytes:
            with path.open("r", encoding="utf-8", newline="") as source:
                return cls(source.read())
        return cls(FileBackedText.from_path(
            path, block_chars=block_chars, cache_blocks=cache_blocks
        ))

    @property
    def storage_backend(self) -> str:
        return "file" if isinstance(self.text, FileBackedText) else "memory"

    def iter_lines(self) -> Iterator[str]:
        if isinstance(self.text, FileBackedText):
            yield from self.text.iter_lines()
        else:
            yield from self.text.splitlines()
