"""Persistent indexing, document metadata, and bounded DRAM cache contracts."""

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from document_ingestion import convert_pdf
from layer1_input import MountedContext, describe
from layer2_controller.code_generator import CodeGenerator
from layer5_context_access import ContextIndex
from main import build_system, default_config_path
from shared.resource_errors import ResourceExhaustedError, resource_error

pytestmark = pytest.mark.indexing
SOURCE = Path(__file__).parent / "fixtures" / "pdf1" / "Mathematics_of_LLMs_2026.pdf"


def mount(tmp_path, text, *, disk=True, **kwargs):
    path = tmp_path / "document.txt"
    path.write_bytes(text.encode("utf-8"))
    return path, MountedContext.from_file(
        path, memory_threshold_bytes=0 if disk else 1_048_576, **kwargs
    )


@pytest.mark.parametrize("disk", [False, True], ids=["ram", "disk"])
def test_metadata_has_exact_counts_character_offsets_and_bounded_previews(tmp_path, disk):
    text = "--- Page 1 ---\n# Overview\ncafé 世界\n\n--- Page 2 ---\n2 Methods\nalpha beta"
    _, ctx = mount(tmp_path, text, disk=disk, block_chars=3)
    metadata = describe(ctx)
    assert metadata.character_count == len(text)
    assert metadata.byte_count == len(text.encode("utf-8"))
    assert metadata.word_count == len(text.split())
    assert metadata.estimated_tokens == (len(text) + 3) // 4
    assert metadata.page_count == 2
    assert [marker.number for marker in metadata.page_markers] == [1, 2]
    assert metadata.page_markers[1].start_char == text.index("--- Page 2 ---")
    assert metadata.heading_count == 2
    assert [heading.title for heading in metadata.headings] == ["Overview", "2 Methods"]
    assert metadata.headings[0].start_char == text.index("# Overview")


def test_metadata_preview_cap_does_not_truncate_actual_counts(tmp_path):
    text = "".join(f"--- Page {number} ---\n# Heading {number}\n" for number in range(1, 51))
    _, ctx = mount(tmp_path, text)
    metadata = describe(ctx)
    assert metadata.page_count == metadata.heading_count == 50
    assert len(metadata.page_markers) == len(metadata.headings) == 32


def test_metadata_handles_markers_and_words_crossing_scan_boundaries(tmp_path):
    text = "x" * 65_530 + "\n--- Page 1 ---\n# Heading\nword " + "z" * 70_000
    _, ctx = mount(tmp_path, text)
    metadata = describe(ctx)
    assert metadata.word_count == len(text.split())
    assert metadata.page_count == 1
    assert metadata.page_markers[0].start_char == text.index("--- Page 1 ---")
    assert metadata.headings[0].title == "Heading"


def test_metadata_ignores_fake_page_markers_and_headings_inside_code(tmp_path):
    _, ctx = mount(tmp_path, "```text\n--- Page 99 ---\n# Not a heading\n```\n# Real\n")
    metadata = describe(ctx)
    assert metadata.has_code_fence
    assert metadata.page_count == 0
    assert metadata.heading_count == 1
    assert metadata.headings[0].title == "Real"


@pytest.mark.parametrize("disk", [False, True], ids=["ram", "disk"])
def test_index_returns_ranked_bounded_matches_with_source_spans(tmp_path, disk):
    text = "--- Page 1 ---\nalpha topic\n\n--- Page 2 ---\nrotary position embedding\n"
    _, ctx = mount(tmp_path, text, disk=disk)
    index = ContextIndex.build(ctx, tmp_path / "indexes", chunk_chars=40, overlap=20)
    hits = index.search("rotary embedding", limit=2)
    assert hits
    assert len(hits) <= 2
    for hit in hits:
        span = text[hit["start_char"]:hit["end_char"]]
        assert "rotary" in span and "embedding" in span
        assert hit["page_start"] <= 2 <= hit["page_end"]
        assert hit["start_line"] == text[:hit["start_char"]].count("\n") + 1
        assert hit["end_line"] == hit["start_line"] + span.count("\n")
        assert len(hit["snippet"]) <= 512
    assert index.metadata["structure"] == json.loads(json.dumps(asdict(describe(ctx))))


def test_index_tracks_pages_beyond_metadata_preview_limit(tmp_path):
    text = "".join(f"--- Page {number} ---\nunique{number}\n" for number in range(1, 51))
    _, ctx = mount(tmp_path, text)
    index = ContextIndex.build(ctx, tmp_path / "indexes", chunk_chars=80, overlap=40)
    hits = index.search("unique49")
    assert hits
    assert any(hit["page_start"] <= 49 <= hit["page_end"] for hit in hits)
    assert len(index.metadata["structure"]["page_markers"]) == 32
    assert index.metadata["structure"]["page_count"] == 50


def test_index_reuse_avoids_rebuilding_fts_tables(tmp_path, monkeypatch):
    _, ctx = mount(tmp_path, "alpha beta gamma")
    first = ContextIndex.build(ctx, tmp_path / "indexes")
    original_mtime = first.path.stat().st_mtime_ns

    def unexpected_scan(*args, **kwargs):
        pytest.fail("Reusing an existing index must not rebuild its metadata or FTS tables")

    monkeypatch.setattr("layer5_context_access.context_index.scan_metadata", unexpected_scan)
    second = ContextIndex.build(ctx, tmp_path / "indexes")
    assert second.path == first.path
    assert second.path.stat().st_mtime_ns == original_mtime
    assert second.search("alpha")


def test_source_and_chunk_configuration_invalidate_persistent_index(tmp_path):
    path, first_ctx = mount(tmp_path, "original alpha")
    root = tmp_path / "indexes"
    first = ContextIndex.build(first_ctx, root)
    different_chunks = ContextIndex.build(first_ctx, root, chunk_chars=128, overlap=16)
    assert first.path != different_chunks.path
    path.write_text("replacement beta", encoding="utf-8")
    replacement = ContextIndex.build(MountedContext.from_file(path), root)
    assert replacement.path != first.path
    assert replacement.search("beta")


def test_hot_result_cache_avoids_sqlite_and_returns_independent_objects(tmp_path, monkeypatch):
    _, ctx = mount(tmp_path, "alpha beta gamma")
    index = ContextIndex.build(ctx, tmp_path / "indexes", dram_cache_bytes=8192)
    original = index.search("alpha")

    def no_connection(*args, **kwargs):
        pytest.fail("A hot-result cache hit must avoid opening SQLite")

    monkeypatch.setattr(sqlite3, "connect", no_connection)
    hot = index.search("alpha")
    assert hot == original
    hot[0]["snippet"] = "caller mutation"
    assert index.search("alpha")[0]["snippet"] != "caller mutation"
    assert index.cache_info["hits"] == 2
    assert index.cache_info["misses"] == 1


def test_result_cache_budget_eviction_and_disable(tmp_path):
    _, ctx = mount(tmp_path, "alpha beta gamma")
    index = ContextIndex.build(ctx, tmp_path / "indexes", dram_cache_bytes=1600)
    for term in ["alpha", "beta", "gamma"]:
        assert index.search(term)
        assert index.cache_info["resident_estimate_bytes"] <= 800
    assert 0 < index.cache_info["entries"] < 3
    disabled = ContextIndex.build(ctx, tmp_path / "indexes", dram_cache_bytes=0)
    assert disabled.search("alpha")
    assert disabled.cache_info["entries"] == 0
    assert disabled.cache_info["sqlite_budget_bytes"] == 0


def test_empty_results_do_not_allow_unbounded_query_key_memory(tmp_path):
    _, ctx = mount(tmp_path, "alpha")
    index = ContextIndex.build(ctx, tmp_path / "indexes", dram_cache_bytes=4000)
    for number in range(40):
        assert index.search(f"missing{number}" + "x" * 300) == []
        assert index.cache_info["resident_estimate_bytes"] <= 2000
    assert index.cache_info["entries"] < 10


def test_block_cache_byte_budget_and_hot_reads(tmp_path):
    _, ctx = mount(
        tmp_path, "abcd" * 100, block_chars=4, cache_blocks=100, max_cache_bytes=440
    )
    assert ctx.text[:4] == ctx.text[:4] == "abcd"
    assert ctx.text.cache_info["hits"] >= 1
    for start in range(0, 400, 4):
        assert ctx.text[start:start + 4] == "abcd"
        assert ctx.text.cache_info["bytes"] <= 440
    assert ctx.text.cache_info["blocks"] <= 2


def test_block_larger_than_dram_budget_is_read_but_not_retained(tmp_path):
    _, ctx = mount(tmp_path, "a" * 100, block_chars=100, max_cache_bytes=10)
    assert ctx.text[:3] == "aaa"
    assert ctx.text.cache_info["blocks"] == 0


def test_stale_source_is_rejected_even_on_cached_search(tmp_path):
    path, ctx = mount(tmp_path, "original alpha")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    assert index.search("alpha")
    path.write_text("changed source content", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        index.search("alpha")


@pytest.mark.parametrize("query", ["", "!!!", "_"])
def test_empty_search_returns_no_results(tmp_path, query):
    _, ctx = mount(tmp_path, "")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    assert index.search(query) == []
    assert index.metadata["chunk_count"] == 0


@pytest.mark.parametrize(("query", "limit"), [
    ("alpha", 0), ("alpha", 101), ("x" * 1001, 5), ("a " * 65, 5),
])
def test_search_rejects_unbounded_query_settings(tmp_path, query, limit):
    _, ctx = mount(tmp_path, "alpha")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    with pytest.raises(ValueError):
        index.search(query, limit=limit)


def test_query_operators_are_literal_not_executable_fts_syntax(tmp_path):
    _, ctx = mount(tmp_path, 'alpha DROP TABLE chunks beta')
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    assert index.search('alpha" DROP TABLE chunks; --')
    assert index.search("beta")


def test_pdf_camelcase_spacing_is_normalized_only_in_index(tmp_path):
    text = "The most widely adopted isRotary\nPosition Embedding (RoPE)."
    _, ctx = mount(tmp_path, text)
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    hits = index.search("Rotary Position Embedding")
    assert hits
    assert "is Rotary" in hits[0]["snippet"]
    assert ctx.text[:] == text
    assert ctx.text[hits[0]["start_char"]:hits[0]["end_char"]] == text


def test_normalization_preserves_case_insensitive_acronym_search(tmp_path):
    _, ctx = mount(tmp_path, "isRotary Position Embedding (RoPE) and ALiBi.")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    for query in ["RoPE", "rope", "ALiBi", "alibi", "isRotary", "rotary"]:
        assert index.search(query), query


@pytest.mark.parametrize("settings", [
    {"chunk_chars": 0}, {"overlap": -1}, {"chunk_chars": 10, "overlap": 10},
    {"dram_cache_bytes": -1},
])
def test_invalid_index_settings_fail_without_artifacts(tmp_path, settings):
    _, ctx = mount(tmp_path, "alpha")
    with pytest.raises(ValueError):
        ContextIndex.build(ctx, tmp_path / "indexes", **settings)
    assert not (tmp_path / "indexes").exists()


def test_corrupted_index_is_not_silently_reused(tmp_path):
    _, ctx = mount(tmp_path, "alpha")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    index.path.write_bytes(b"broken database")
    with pytest.raises(ValueError, match="Invalid context index"):
        ContextIndex.build(ctx, tmp_path / "indexes")


def test_missing_index_is_reported_as_rebuild_error(tmp_path):
    _, ctx = mount(tmp_path, "alpha")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    index.path.unlink()
    with pytest.raises(ValueError, match="rebuild"):
        index.search("alpha")


def test_tools_refuse_an_index_for_a_different_mounted_context(tmp_path):
    _, ctx = mount(tmp_path, "alpha")
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    controller, _ = build_system(default_config_path(), context_index=index)
    controller.runtime.state.set("context", "different document")
    result = controller.runtime.execute("result = search_context('alpha')")
    assert not result.ok
    assert "does not match" in result.error


@pytest.mark.parametrize(("message", "code"), [
    ("database or disk is full", "disk_full"), ("out of memory", "out_of_memory"),
])
def test_sqlite_resource_errors_preserve_stable_codes(tmp_path, monkeypatch, message, code):
    _, ctx = mount(tmp_path, "alpha")

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(sqlite3, "connect", fail)
    with pytest.raises(ResourceExhaustedError) as caught:
        ContextIndex.build(ctx, tmp_path / "indexes")
    assert caught.value.code == code
    assert not list((tmp_path / "indexes").iterdir())
    assert resource_error(sqlite3.OperationalError("no such table")) is None


def test_pdf1_index_metadata_page_spans_and_existing_repl(tmp_path):
    artifact = convert_pdf(SOURCE, tmp_path / "pdf-cache", ocr_enabled=False)
    ctx = MountedContext.from_file(artifact.text_path, memory_threshold_bytes=0)
    index = ContextIndex.build(ctx, tmp_path / "indexes")
    matches = index.search("Rotary Position Embedding")
    assert matches
    assert any(hit["page_start"] <= 14 <= hit["page_end"] for hit in matches)
    body = next(hit for hit in matches if hit["page_start"] <= 14 <= hit["page_end"])
    assert index.metadata["structure"]["page_count"] == 20
    controller, _ = build_system(default_config_path(), context_index=index)
    code = (
        "matches = search_context('Rotary Position Embedding')\n"
        "metadata = context_info()\n"
        "for match in matches:\n"
        "    if match['page_start'] <= 14 <= match['page_end']:\n"
        "        source_read = read_context(match['start_char'], match['end_char'])\n"
        "        result = source_read['text']\n"
        "        break"
    )
    responses = iter([
        json.dumps({"action": "execute", "code": code}),
        json.dumps({
            "action": "finalize_variable", "name": "result",
            "evidence": [{
                "document_hash": index.metadata["source_sha256"],
                "start_char": body["start_char"], "end_char": body["end_char"],
            }],
        }),
    ])
    controller.codegen = CodeGenerator(chat=lambda *args, **kwargs: (next(responses), 0))
    answer = controller.run_until_done(ctx, "Where is RoPE explained?")
    assert "Rotary" in answer.require_answer() and "RoPE" in answer.require_answer()
    assert answer.verification == "provenance_validated"
    assert controller.runtime.state.get("metadata")["page_count"] == 20
    assert not ctx.text._cache or ctx.text.cache_info["bytes"] <= ctx.text.max_cache_bytes


def test_pdf_manifest_has_exact_utf8_page_offsets_and_document_statistics(tmp_path, monkeypatch):
    from document_ingestion.text_extractor import ExtractedPage

    def pages(*args, **kwargs):
        kwargs["document_info"]["title"] = "Fixture title"
        yield ExtractedPage(1, "café 世界", True)
        yield ExtractedPage(2, "second page", True)

    monkeypatch.setattr("document_ingestion.pdf_to_text.extract_pages", pages)
    artifact = convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    text = artifact.text_path.read_text(encoding="utf-8")
    data = artifact.text_path.read_bytes()
    for page, expected in zip(manifest["pages"], ["café 世界", "second page"]):
        assert text[page["char_start"]:page["char_end"]] == expected
        assert data[page["byte_start"]:page["byte_end"]].decode("utf-8") == expected
        assert page["word_count"] == len(expected.split())
    assert manifest["source"]["pdf_metadata"]["title"] == "Fixture title"
    assert manifest["source"]["byte_count"] == SOURCE.stat().st_size
    assert manifest["text_metadata"]["character_count"] == len(text)
    assert manifest["text_metadata"]["byte_count"] == len(data)
    assert manifest["text_metadata"]["page_count"] == 2
    assert manifest["created_at"].endswith("+00:00")


def test_cli_local_index_search_and_metadata_do_not_require_llm(tmp_path):
    path, _ = mount(tmp_path, "--- Page 1 ---\n# Heading\nrotary position embedding\n")
    common = [
        sys.executable, "-m", "main", "--context-file", str(path),
        "--index-dir", str(tmp_path / "indexes"), "--memory-threshold-bytes", "0",
    ]
    search = subprocess.run(common + ["--search", "rotary embedding"], capture_output=True, text=True, timeout=30)
    assert search.returncode == 0, search.stderr
    assert json.loads(search.stdout)["matches"]
    metadata = subprocess.run(common + ["--index", "--show-metadata"], capture_output=True, text=True, timeout=30)
    assert metadata.returncode == 0, metadata.stderr
    info = json.loads(metadata.stdout)
    assert info["storage_backend"] == "file"
    assert info["structure"]["heading_count"] == 1
    assert info["index"]["chunk_count"] == 1
