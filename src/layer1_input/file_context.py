"""Read-only UTF-8 text with character indexing and a bounded block cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Iterator
from shared.resource_errors import report_resource_errors


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _cache_cost(index: int, text: str) -> int:
    return sys.getsizeof(index) + sys.getsizeof(text) + 128


@dataclass(frozen=True)
class FileBackedText:
    """String-like slicing without retaining the complete document in RAM.

    A streaming indexing pass records TextIO seek cookies every ``block_chars``
    characters. Only requested blocks are cached. Sources must remain unchanged
    for the lifetime of the context; changes are rejected rather than silently
    mixing old cached text with new file contents.
    """

    _path: Path = field(repr=False)
    _length: int
    _checkpoints: tuple[int, ...] = field(repr=False)
    _source_fingerprint: tuple[int, int, int, int] = field(repr=False)
    block_chars: int = 65_536
    cache_blocks: int = 8
    max_cache_bytes: int = 4_194_304
    _cache: OrderedDict[int, str] = field(
        default_factory=OrderedDict, repr=False, compare=False, init=False
    )
    _cache_stats: dict[str, int] = field(
        default_factory=lambda: {"hits": 0, "misses": 0},
        repr=False, compare=False, init=False,
    )

    @classmethod
    @report_resource_errors
    def from_path(
        cls, path: str | Path, *, block_chars: int = 65_536, cache_blocks: int = 8,
        max_cache_bytes: int = 4_194_304,
    ) -> "FileBackedText":
        if block_chars <= 0 or cache_blocks < 0 or max_cache_bytes < 0:
            raise ValueError("block_chars must be positive; cache_blocks and max_cache_bytes must be non-negative")
        path = Path(path).resolve(strict=True)
        before = _fingerprint(path)
        checkpoints: list[int] = []
        length = 0
        # Keep original line endings so character offsets are stable.
        with path.open("r", encoding="utf-8", newline="") as source:
            while True:
                cookie = source.tell()
                block = source.read(block_chars)
                if not block:
                    break
                checkpoints.append(cookie)
                length += len(block)
        if _fingerprint(path) != before:
            raise RuntimeError("context file changed during indexing")
        return cls(path, length, tuple(checkpoints), before, block_chars, cache_blocks, max_cache_bytes)

    def __len__(self) -> int:
        self._check_source()
        return self._length

    def __repr__(self) -> str:
        return f"FileBackedText(characters={self._length}, cache_blocks={self.cache_blocks})"

    def __str__(self) -> str:
        # Never implicitly read the whole source through str() or an f-string.
        return repr(self)

    def __reduce__(self):
        # Worker IPC transfers the index, not cached document text.
        return (
            type(self),
            (
                self._path, self._length, self._checkpoints,
                self._source_fingerprint, self.block_chars, self.cache_blocks, self.max_cache_bytes,
            ),
        )

    def _check_source(self) -> None:
        if _fingerprint(self._path) != self._source_fingerprint:
            raise RuntimeError("context file changed after mounting")

    def _block(self, index: int) -> str:
        self._check_source()
        if index in self._cache:
            self._cache_stats["hits"] += 1
            self._cache.move_to_end(index)
            return self._cache[index]
        self._cache_stats["misses"] += 1
        with self._path.open("r", encoding="utf-8", newline="") as source:
            source.seek(self._checkpoints[index])
            block = source.read(self.block_chars)
        self._check_source()
        cost = _cache_cost(index, block)
        if self.cache_blocks and cost <= self.max_cache_bytes:
            while self._cache and (
                len(self._cache) >= self.cache_blocks
                or sum(_cache_cost(key, value) for key, value in self._cache.items()) + cost > self.max_cache_bytes
            ):
                self._cache.popitem(last=False)
            self._cache[index] = block
        return block

    @property
    def cache_info(self) -> dict[str, int]:
        return {
            **self._cache_stats, "blocks": len(self._cache),
            "bytes": sum(_cache_cost(key, value) for key, value in self._cache.items()),
            "max_bytes": self.max_cache_bytes,
        }

    def _read_span(self, start: int, stop: int) -> str:
        if start >= stop:
            return ""
        parts = []
        for index in range(start // self.block_chars, (stop - 1) // self.block_chars + 1):
            offset = index * self.block_chars
            parts.append(self._block(index)[max(0, start - offset):stop - offset])
        return "".join(parts)

    @report_resource_errors
    def __getitem__(self, key: int | slice) -> str:
        self._check_source()
        if isinstance(key, int):
            index = key if key >= 0 else self._length + key
            if not 0 <= index < self._length:
                raise IndexError("context index out of range")
            return self._read_span(index, index + 1)
        if not isinstance(key, slice):
            raise TypeError("context indices must be integers or slices")
        start, stop, step = key.indices(self._length)
        if step == 1:
            return self._read_span(start, stop)
        indices = range(start, stop, step)
        if not indices:
            return ""
        low, high = min(indices[0], indices[-1]), max(indices[0], indices[-1])
        span = self._read_span(low, high + 1)
        return span[::step] if step > 0 else span[::-1][:: -step]

    def iter_lines(self) -> Iterator[str]:
        """Stream lines (memory use is bounded by the longest source line)."""
        self._check_source()
        with self._path.open("r", encoding="utf-8", newline="") as source:
            for line in source:
                self._check_source()
                yield from line.splitlines()
        self._check_source()

    @property
    def byte_length(self) -> int:
        self._check_source()
        return self._source_fingerprint[2]
