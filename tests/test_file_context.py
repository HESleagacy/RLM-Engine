import pickle
import subprocess
import sys

import pytest

from layer1_input import FileBackedText, MountedContext, byte_length, describe

pytestmark = pytest.mark.storage


def mount(tmp_path, text, **kwargs):
    path = tmp_path / "document.txt"
    path.write_bytes(text.encode("utf-8"))
    return path, MountedContext.from_file(path, **kwargs)


def test_small_files_remain_in_memory_and_preserve_whitespace(tmp_path):
    text = "  café\r\n世界\n\n"
    path, ctx = mount(tmp_path, text)
    assert ctx.storage_backend == "memory"
    assert ctx.text == text
    path.unlink()
    assert ctx.text == text


def test_threshold_is_based_on_bytes_not_characters(tmp_path):
    _, ctx = mount(tmp_path, "éé", memory_threshold_bytes=3)
    assert ctx.storage_backend == "file"
    assert len(ctx.text) == 2
    assert byte_length(ctx) == 4


@pytest.mark.parametrize("block_chars", [1, 2, 7, 64])
def test_file_slicing_matches_python_strings(tmp_path, block_chars):
    text = "α🙂\r\n世界 café\nend"
    _, ctx = mount(
        tmp_path, text, memory_threshold_bytes=0, block_chars=block_chars
    )
    assert isinstance(ctx.text, FileBackedText)
    assert len(ctx.text) == len(text)
    for index in range(-len(text), len(text)):
        assert ctx.text[index] == text[index]
    for start in [None, -100, -5, 0, 2, 100]:
        for stop in [None, -100, -2, 0, 5, 100]:
            for step in [None, 1, 2, -1, -3]:
                selection = slice(start, stop, step)
                assert ctx.text[selection] == text[selection]
    with pytest.raises(IndexError):
        _ = ctx.text[len(text)]
    with pytest.raises(ValueError):
        _ = ctx.text[::0]


def test_serialization_preserves_file_descriptor_but_drops_cached_text(tmp_path):
    _, ctx = mount(
        tmp_path, "abcd" * 100, memory_threshold_bytes=0,
        block_chars=4, cache_blocks=2,
    )
    text = ctx.text
    assert text[:4] == "abcd"
    restored = pickle.loads(pickle.dumps(text))
    assert not restored._cache
    assert restored[:12] == "abcd" * 3


def test_cache_can_be_disabled(tmp_path):
    _, ctx = mount(tmp_path, "abcdef", memory_threshold_bytes=0, cache_blocks=0)
    assert ctx.text[:3] == "abc"
    assert not ctx.text._cache


@pytest.mark.parametrize("text", [
    "", "\n", "a\n", "\n\n", " a\r\n\r\nb\n", "a\rb",
    "a\n \t\n b", " \n\n ", "a\n\nb\n\nc", "```python\nx = 1\n```",
    "a\u2028b\n\nc",
])
def test_streamed_metadata_and_lines_match_memory(tmp_path, text):
    path, _ = mount(tmp_path, text)
    # Construct the disk backend directly to exercise empty files as well.
    disk = MountedContext(FileBackedText.from_path(path, block_chars=1))
    memory = MountedContext(text)
    assert list(disk.iter_lines()) == list(memory.iter_lines())
    assert describe(disk) == describe(memory)
    assert byte_length(disk) == byte_length(memory)


@pytest.mark.parametrize("kwargs", [
    {"memory_threshold_bytes": -1},
    {"block_chars": 0},
    {"cache_blocks": -1},
])
def test_invalid_storage_settings_are_rejected(tmp_path, kwargs):
    path, _ = mount(tmp_path, "test")
    with pytest.raises(ValueError):
        MountedContext.from_file(path, **kwargs)


@pytest.mark.parametrize("threshold", [0, 100])
def test_invalid_utf8_is_rejected(tmp_path, threshold):
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        MountedContext.from_file(path, memory_threshold_bytes=threshold)


def test_cli_rejects_conflicting_context_sources(tmp_path):
    path, _ = mount(tmp_path, "Hello from disk")
    conflict = subprocess.run([
        sys.executable, "-m", "main", "--context-file", str(path), "--prompt", "other",
    ], capture_output=True, text=True, timeout=30)
    assert conflict.returncode == 2
    assert "not allowed with argument" in conflict.stderr
