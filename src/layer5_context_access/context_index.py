"""Persistent local full-text indexing with bounded hot-result DRAM caching."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict, dataclass, field
import errno
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from layer1_input.context_repr import MountedContext
from layer1_input.metadata_provider import scan_metadata
from shared.resource_errors import report_resource_errors, resource_error

_SCHEMA = 1
_PROSE_BOUNDARY = re.compile(r"\b([a-z]+)(?=[A-Z])")


def _normalize(text: str) -> str:
    # Split "isRotary", but preserve mixed-case names such as RoPE and ALiBi.
    return _PROSE_BOUNDARY.sub(r"\1 ", text)


def _cache_cost(key, payload: bytes) -> int:
    # Include query keys and per-entry bookkeeping, not only result JSON.
    return sum(sys.getsizeof(item) for item in (key, key[0], key[1], payload)) + 128


def _digest(ctx: MountedContext) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(ctx.text), 65_536):
        digest.update(ctx.text[start:start + 65_536].encode("utf-8"))
    return digest.hexdigest()


def _connect(path: Path, cache_bytes: int, *, readonly: bool = False):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro" if readonly else str(path), uri=readonly)
    try:
        connection.execute(f"PRAGMA cache_size = {-max(0, cache_bytes // 1024)}")
        connection.execute("PRAGMA mmap_size = 0")
        return connection
    except Exception:
        connection.close()
        raise


def _read_info(path: Path, key: str, cache_bytes: int) -> dict:
    try:
        with closing(_connect(path, cache_bytes, readonly=True)) as database:
            row = database.execute("SELECT value FROM metadata WHERE name = 'document'").fetchone()
            info = json.loads(row[0])
            count = database.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if info["index_key"] != key or info["schema"] != _SCHEMA or info["chunk_count"] != count:
                raise ValueError("Index header does not match its contents")
            return info
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as exc:
        if resource_error(exc) is not None:
            raise
        raise ValueError(f"Invalid context index at {path}; remove that generated index and rebuild") from exc


@dataclass
class ContextIndex:
    path: Path
    context: MountedContext = field(repr=False)
    metadata: dict
    dram_cache_bytes: int = 4_194_304
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False, init=False)
    _stats: dict = field(default_factory=lambda: {"hits": 0, "misses": 0}, repr=False, init=False)

    @classmethod
    @report_resource_errors
    def build(
        cls, ctx: MountedContext, directory: str | Path, *,
        chunk_chars: int = 2048, overlap: int = 128, dram_cache_bytes: int = 4_194_304,
    ) -> "ContextIndex":
        if chunk_chars <= 0 or not 0 <= overlap < chunk_chars or dram_cache_bytes < 0:
            raise ValueError("chunk_chars must be positive, 0 <= overlap < chunk_chars, and DRAM cache bytes non-negative")
        settings = {
            "schema": _SCHEMA, "source_sha256": _digest(ctx),
            "chunk_chars": chunk_chars, "overlap": overlap,
            "normalization": "ascii-lowercase-prefix-boundaries-v2",
        }
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        root = Path(directory).expanduser().resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        final = root / key
        path = final / "index.sqlite3"
        if final.exists():
            return cls(path, ctx, _read_info(path, key, dram_cache_bytes // 2), dram_cache_bytes)
        with tempfile.TemporaryDirectory(prefix=".indexing-", dir=root) as work:
            temporary = Path(work)
            with closing(_connect(temporary / "index.sqlite3", dram_cache_bytes // 2)) as database:
                try:
                    database.execute("CREATE VIRTUAL TABLE search USING fts5(text, tokenize='unicode61')")
                except sqlite3.OperationalError as exc:
                    if "no such module" in str(exc).lower():
                        raise RuntimeError("This Python SQLite build requires FTS5 support for indexing") from exc
                    raise
                database.executescript("""
                    CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE pages(start_char INTEGER PRIMARY KEY, number INTEGER NOT NULL);
                    CREATE TABLE chunks(
                        id INTEGER PRIMARY KEY, start_char INTEGER NOT NULL,
                        end_char INTEGER NOT NULL, start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL
                    );
                """)
                structure = scan_metadata(
                    ctx, on_page=lambda page: database.execute(
                        "INSERT INTO pages VALUES (?, ?)", (page.start_char, page.number)
                    ),
                )
                step, line, count = chunk_chars - overlap, 1, 0
                for start in range(0, len(ctx.text), step):
                    text = ctx.text[start:start + chunk_chars]
                    count += 1
                    # PDFs can concatenate prose and capitalized terms ("isRotary").
                    # Normalize index text only; source spans still address exact input.
                    indexed_text = _normalize(text)
                    database.execute("INSERT INTO search(rowid, text) VALUES (?, ?)", (count, indexed_text))
                    database.execute(
                        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                        (count, start, start + len(text), line, line + text.count("\n")),
                    )
                    line += text[:step].count("\n")
                info = {
                    **settings, "index_key": key, "chunk_count": count,
                    "structure": asdict(structure), "search_type": "literal-token AND (SQLite FTS5)",
                }
                database.execute("INSERT INTO metadata VALUES ('document', ?)", (json.dumps(info),))
                database.commit()
            len(ctx.text)  # Reject changed/unmounted file sources before publication.
            try:
                temporary.rename(final)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
        return cls(path, ctx, _read_info(path, key, dram_cache_bytes // 2), dram_cache_bytes)

    @property
    def cache_info(self) -> dict:
        return {
            **self._stats, "entries": len(self._cache),
            "result_bytes": sum(len(value) for value in self._cache.values()),
            "resident_estimate_bytes": sum(_cache_cost(key, value) for key, value in self._cache.items()),
            "result_budget_bytes": self.dram_cache_bytes // 2,
            "sqlite_budget_bytes": self.dram_cache_bytes // 2,
        }

    @report_resource_errors
    def search(self, query: str, *, limit: int = 5) -> list[dict]:
        len(self.context.text)  # Check source identity even on a hot-cache hit.
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("Search query must be a string of at most 1000 characters")
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100")
        terms = re.findall(r"[^\W_]+", _normalize(query))
        if not terms:
            return []
        if len(terms) > 64:
            raise ValueError("Search query must contain at most 64 terms")
        expression = " AND ".join(f'"{term}"' for term in terms)
        key = (expression, limit)
        if key in self._cache:
            self._stats["hits"] += 1
            self._cache.move_to_end(key)
            return json.loads(self._cache[key])
        self._stats["misses"] += 1
        try:
            with closing(_connect(self.path, self.dram_cache_bytes // 2, readonly=True)) as database:
                rows = database.execute("""
                    SELECT c.id, c.start_char, c.end_char, c.start_line, c.end_line,
                        (SELECT number FROM pages WHERE start_char <= c.start_char ORDER BY start_char DESC LIMIT 1),
                        (SELECT number FROM pages WHERE start_char < c.end_char ORDER BY start_char DESC LIMIT 1),
                        bm25(search), snippet(search, 0, '', '', ' … ', 32)
                    FROM search JOIN chunks c ON c.id = search.rowid
                    WHERE search MATCH ? ORDER BY bm25(search), c.id LIMIT ?
                """, (expression, limit)).fetchall()
        except sqlite3.Error as exc:
            if resource_error(exc) is not None:
                raise
            raise ValueError("Cannot search the context index; rebuild the generated index") from exc
        results = [
            {
                "chunk_id": row[0], "start_char": row[1], "end_char": row[2],
                "start_line": row[3], "end_line": row[4], "page_start": row[5],
                "page_end": row[6], "score": -row[7], "snippet": row[8][:512],
            }
            for row in rows
        ]
        payload = json.dumps(results, ensure_ascii=False).encode("utf-8")
        budget = self.dram_cache_bytes // 2
        cost = _cache_cost(key, payload)
        if cost <= budget:
            while self._cache and (
                len(self._cache) >= 1024
                or sum(_cache_cost(key, value) for key, value in self._cache.items()) + cost > budget
            ):
                self._cache.popitem(last=False)
            self._cache[key] = payload
        return results
