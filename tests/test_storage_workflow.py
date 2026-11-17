"""Storage contract tests: real files, instrumented I/O, and offline execution."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import os
import hashlib
import json
import subprocess
import sys

import pytest

from layer1_input import FileBackedText, MountedContext, byte_length
from layer2_controller.code_generator import CodeGenerator
from layer5_context_access import by_keyword, by_regex, iter_windows, peek_head, peek_tail
from main import build_system, default_config_path

pytestmark = pytest.mark.storage


@pytest.fixture
def file_reads(monkeypatch):
    """Observe Python-level source reads without relying on RSS or timing."""
    original_open = Path.open

    class ReadLog:
        path = None
        after_read = None

        def __init__(self):
            self.sizes = []
            self.lines = []

        def clear(self):
            self.sizes.clear()
            self.lines.clear()

    log = ReadLog()

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            log.sizes.append(size)
            value = self.stream.read(size)
            if log.after_read is not None:
                callback, log.after_read = log.after_read, None
                callback()
            return value

        def __iter__(self):
            for line in self.stream:
                log.lines.append(len(line))
                yield line

    def tracked_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if log.path is not None and path.resolve() == log.path.resolve() and mode == "r":
            return Reader(stream)
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)
    return log


@pytest.mark.parametrize(("size", "expected"), [
    (0, "memory"), (1_048_575, "memory"),
    (1_048_576, "memory"), (1_048_577, "file"),
])
def test_default_threshold_boundary(tmp_path, size, expected):
    path = tmp_path / "document.txt"
    path.write_bytes(b"a" * size)
    ctx = MountedContext.from_file(path)
    assert ctx.storage_backend == expected
    assert len(ctx.text) == size
    assert ctx.text[:12] == "a" * min(size, 12)


def test_large_file_indexing_and_reads_never_request_unbounded_text(tmp_path, file_reads):
    path = tmp_path / "large.txt"
    # Write incrementally so even the fixture does not require a full document string.
    with path.open("wb") as stream:
        for _ in range(64):
            stream.write(b"abcdefgh" * 8192)
    file_reads.path = path
    ctx = MountedContext.from_file(path, block_chars=4096, cache_blocks=2)
    assert ctx.storage_backend == "file"
    assert not ctx.text._cache
    assert len(ctx.text._checkpoints) == path.stat().st_size // 4096
    assert file_reads.sizes and set(file_reads.sizes) == {4096}

    file_reads.clear()
    assert ctx.text[-8:] == "abcdefgh"
    assert file_reads.sizes == [4096]
    assert len(ctx.text._cache) == 1


def test_cache_hits_avoid_disk_reads_and_evicted_blocks_are_reloaded(tmp_path, file_reads):
    path = tmp_path / "cache.txt"
    path.write_text("aaaabbbbcccc", encoding="utf-8")
    file_reads.path = path
    ctx = MountedContext.from_file(
        path, memory_threshold_bytes=0, block_chars=4, cache_blocks=2
    )
    file_reads.clear()
    for selection, expected in [
        (slice(0, 4), "aaaa"), (slice(4, 8), "bbbb"),
        (slice(0, 4), "aaaa"), (slice(8, 12), "cccc"),
    ]:
        assert ctx.text[selection] == expected
    assert file_reads.sizes == [4, 4, 4]  # The repeated first block was cached.
    assert list(ctx.text._cache) == [0, 2]  # Block 1, not block 0, was evicted.
    assert ctx.text[4:8] == "bbbb"
    assert file_reads.sizes == [4, 4, 4, 4]
    assert len(ctx.text._cache) == 2


def test_descriptor_and_size_metadata_do_not_read_document_contents(tmp_path, file_reads):
    path = tmp_path / "private.txt"
    path.write_text("CONFIDENTIAL DOCUMENT CONTENT", encoding="utf-8")
    ctx = MountedContext.from_file(path, memory_threshold_bytes=0)
    file_reads.path = path
    assert len(ctx.text) == 29
    assert byte_length(ctx) == 29
    assert "CONFIDENTIAL" not in str(ctx.text)
    assert str(path) not in repr(ctx.text)
    assert f"{ctx.text}" == str(ctx.text)
    assert not file_reads.sizes
    assert not file_reads.lines


def test_chunk_iterator_reads_only_when_consumed(tmp_path, file_reads):
    path = tmp_path / "chunks.txt"
    path.write_text("aaaabbbbccccdddd", encoding="utf-8")
    ctx = MountedContext.from_file(
        path, memory_threshold_bytes=0, block_chars=4, cache_blocks=1
    )
    file_reads.path = path
    chunks = iter_windows(ctx, size=4)
    assert not file_reads.sizes
    assert next(chunks) == "aaaa"
    assert file_reads.sizes == [4]
    assert next(chunks) == "bbbb"
    assert file_reads.sizes == [4, 4]
    chunks.close()
    assert file_reads.sizes == [4, 4]  # Remaining chunks were never requested.


@pytest.mark.parametrize("backend", ["memory", "file"])
@pytest.mark.parametrize(("size", "overlap"), [
    (1, 0), (5, 0), (5, 2), (5, 4), (100, 0),
])
def test_overlapping_chunks_cover_the_exact_original_text(tmp_path, backend, size, overlap):
    text = "α🙂\r\nlonger text 世界"
    path = tmp_path / "unicode.txt"
    path.write_bytes(text.encode("utf-8"))
    ctx = MountedContext.from_file(
        path, memory_threshold_bytes=0 if backend == "file" else 1000, block_chars=3
    )
    chunks = list(iter_windows(ctx, size, overlap))
    # Each subsequent window extends the already reconstructed prefix.
    reconstructed = chunks[0]
    covered = len(chunks[0])
    for index, chunk in enumerate(chunks[1:], 1):
        start = index * (size - overlap)
        reconstructed += chunk[max(0, covered - start):]
        covered = max(covered, start + len(chunk))
    assert reconstructed == text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (4, -1), (4, 4)])
def test_invalid_chunk_parameters_fail_before_reading(tmp_path, file_reads, size, overlap):
    path = tmp_path / "chunks.txt"
    path.write_text("abcdefgh", encoding="utf-8")
    ctx = MountedContext.from_file(path, memory_threshold_bytes=0)
    file_reads.path = path
    with pytest.raises(ValueError):
        list(iter_windows(ctx, size, overlap))
    assert not file_reads.sizes
    assert not file_reads.lines


@pytest.mark.parametrize("threshold", [0, 1000], ids=["disk", "ram"])
def test_searches_match_in_both_backends_without_full_file_reads(tmp_path, file_reads, threshold):
    path = tmp_path / "search.txt"
    path.write_bytes("Alpha\r\nβeta target\nTARGET again\nend".encode("utf-8"))
    ctx = MountedContext.from_file(path, memory_threshold_bytes=threshold, block_chars=2)
    file_reads.path = path
    assert by_keyword(ctx, "target") == ["βeta target", "TARGET again"]
    assert by_regex(ctx, r"^βeta.*target$") == ["βeta target"]
    assert file_reads.sizes == []
    assert len(file_reads.lines) == (8 if ctx.storage_backend == "file" else 0)
    if ctx.storage_backend == "file":
        assert not ctx.text._cache


def test_source_replacement_during_indexing_is_rejected(tmp_path, file_reads):
    path = tmp_path / "changing.txt"
    path.write_text("original contents", encoding="utf-8")
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("new contents", encoding="utf-8")
    file_reads.path = path
    file_reads.after_read = lambda: replacement.replace(path)
    with pytest.raises(RuntimeError, match="changed during indexing"):
        MountedContext.from_file(path, memory_threshold_bytes=0, block_chars=2)


@pytest.mark.parametrize("change", ["truncate", "replace", "remove"])
def test_cached_reads_do_not_hide_source_changes(tmp_path, change):
    path = tmp_path / "source.txt"
    path.write_text("original", encoding="utf-8")
    ctx = MountedContext.from_file(path, memory_threshold_bytes=0)
    assert ctx.text[:] == "original"  # Populate the cache.
    if change == "truncate":
        path.write_bytes(b"")
    elif change == "replace":
        replacement = tmp_path / "replacement.txt"
        replacement.write_text("replaced", encoding="utf-8")  # Same byte length.
        before = path.stat()
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
    else:
        path.unlink()
    expected = FileNotFoundError if change == "remove" else RuntimeError
    for read in [lambda: len(ctx.text), lambda: ctx.text[:3], lambda: byte_length(ctx)]:
        with pytest.raises(expected):
            read()


def test_source_change_while_streaming_lines_is_rejected(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("first\nsecond\nthird", encoding="utf-8")
    ctx = MountedContext.from_file(path, memory_threshold_bytes=0)
    lines = ctx.iter_lines()
    assert next(lines) == "first"
    path.write_text("replacement", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        next(lines)


@pytest.mark.parametrize("threshold", [0, 1000])
def test_read_only_document_with_spaces_and_unicode_path(tmp_path, threshold):
    directory = tmp_path / "External SSD 世界"
    directory.mkdir()
    path = directory / "my document.txt"
    original = "  Header\r\n世界🙂\nTail  ".encode("utf-8")
    path.write_bytes(original)
    path.chmod(0o444)
    try:
        ctx = MountedContext.from_file(path, memory_threshold_bytes=threshold)
        assert peek_head(ctx, 8) == "  Header"
        assert peek_tail(ctx, 6) == "Tail  "
        assert path.read_bytes() == original
    finally:
        path.chmod(0o600)


def test_mounted_context_and_file_configuration_are_frozen(tmp_path):
    path = tmp_path / "frozen.txt"
    path.write_text("original", encoding="utf-8")
    ctx = MountedContext.from_file(path, memory_threshold_bytes=0)
    with pytest.raises(FrozenInstanceError):
        ctx.text = "replacement"
    with pytest.raises(FrozenInstanceError):
        ctx.text.block_chars = 1
    assert ctx.text[:] == "original"


def test_multiple_repl_rounds_preserve_file_aliases_and_bounded_state(tmp_path):
    path = tmp_path / "rounds.txt"
    path.write_text("alpha\nbeta\nomega", encoding="utf-8")
    ctx = MountedContext.from_file(
        path, memory_threshold_bytes=0, block_chars=4, cache_blocks=1
    )
    controller, monitor = build_system(default_config_path())
    seen_messages = []
    second_code = (
        "count = 0\n"
        "for chunk in iter_chunks(4):\n"
        "    count += len(chunk)\n"
        "hits = by_keyword('beta')\n"
        "matches = by_regex('^omega$')\n"
        "legacy_chunks = chunker(4)\n"
        "result = head + ':' + alias[-5:]\n"
        "print(result)\n"
    )
    responses = iter([
        json.dumps({
            "action": "execute",
            "code": "source = read_context(0, len(context))\nalias = context\nhead = peek_head(5)\nprint(head)",
        }),
        json.dumps({"action": "execute", "code": second_code}),
        json.dumps({
            "action": "finalize_variable", "name": "result",
            "evidence": [{
                "document_hash": hashlib.sha256(b"alpha\nbeta\nomega").hexdigest(),
                "start_char": 0, "end_char": 16,
            }],
        }),
    ])

    def chat(messages, *, max_tokens=None, timeout_seconds=None):
        seen_messages.append(messages)
        return next(responses), 0

    controller.codegen = CodeGenerator(chat=chat)
    assert controller.run_until_done(ctx, "Read both ends").require_answer() == "alpha:omega"
    assert monitor.steps.current == 2
    assert controller.runtime.state.get("count") == len(ctx.text)
    assert controller.runtime.state.get("hits") == ["beta"]
    assert controller.runtime.state.get("matches") == ["omega"]
    assert "".join(controller.runtime.state.get("legacy_chunks")) == "alpha\nbeta\nomega"
    assert "alpha:omega" in seen_messages[-1][-1]["content"]
    alias = controller.runtime.state.get("alias")
    assert isinstance(alias, FileBackedText)
    assert not alias._cache  # IPC sent the descriptor, not cached blocks.
    assert ctx.text.cache_info["blocks"] <= 1
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\nomega"


def test_reused_controller_switches_backends_without_leaking_previous_context(tmp_path):
    controller, _ = build_system(default_config_path())
    contexts = []
    for index, threshold in enumerate([0, 1000, 0]):
        path = tmp_path / f"document-{index}.txt"
        path.write_text(f"document-{index}", encoding="utf-8")
        contexts.append(MountedContext.from_file(path, memory_threshold_bytes=threshold))
    for index, ctx in enumerate(contexts):
        responses = iter([
            json.dumps({
                "action": "execute",
                "code": "source = read_context(0, len(context))\nresult = peek_head(100) + ':' + query",
            }),
            json.dumps({
                "action": "finalize_variable", "name": "result",
                "evidence": [{
                    "document_hash": hashlib.sha256(f"document-{index}".encode()).hexdigest(),
                    "start_char": 0, "end_char": len(f"document-{index}"),
                }],
            }),
        ])
        controller.codegen = CodeGenerator(chat=lambda *args, **kwargs: (next(responses), 0))
        answer = controller.run_until_done(ctx, f"query-{index}")
        assert answer.require_answer() == f"document-{index}:query-{index}"
        assert controller.runtime.state.get("context") is ctx.text
        assert controller.flow.iteration == 2


def test_worker_reports_unavailable_source_without_overwriting_mounted_context(tmp_path):
    path = tmp_path / "removed.txt"
    path.write_text("original", encoding="utf-8")
    ctx = MountedContext.from_file(path, memory_threshold_bytes=0)
    controller, _ = build_system(default_config_path())
    controller.runtime.state.set("context", ctx.text)
    path.unlink()
    result = controller.runtime.execute("result = context[:5]")
    assert not result.ok
    assert "FileNotFoundError" in result.error
    assert result.value is None
    assert controller.runtime.state.get("context") is ctx.text


@pytest.mark.parametrize(("case", "message"), [
    ("missing", "No such file"),
    ("invalid_utf8", "utf-8"),
    ("negative_threshold", "memory_threshold_bytes"),
    ("zero_block", "block_chars"),
    ("negative_cache", "cache_blocks"),
])
def test_cli_storage_errors_are_actionable_and_do_not_initialize_llm(tmp_path, case, message):
    path = tmp_path / "document.txt"
    path.write_text("document", encoding="utf-8")
    flags = []
    if case == "missing":
        path.unlink()
    elif case == "invalid_utf8":
        path.write_bytes(b"\xff")
    else:
        flags = {
            "negative_threshold": ["--memory-threshold-bytes", "-1"],
            "zero_block": ["--context-block-chars", "0"],
            "negative_cache": ["--context-cache-blocks", "-1"],
        }[case]
    # An invalid document must fail before Groq initialization, even without a key.
    env = dict(os.environ, GROQ_API_KEY="")
    result = subprocess.run([
        sys.executable, "-m", "main", "--context-file", str(path), "--use-groq", *flags,
    ], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert "Missing GROQ_API_KEY" not in result.stderr
