"""Real PDF 1 → text artifact → hybrid mount → deterministic REPL retrieval."""

import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

import pytest

from document_ingestion import convert_pdf
from document_ingestion.artifact_store import file_digest
from document_ingestion.text_extractor import PDFIngestionError
from layer1_input import MountedContext, byte_length, describe
from layer2_controller.code_generator import CodeGenerator
from layer5_context_access import by_regex, iter_windows
from main import build_system, default_config_path

pytestmark = [pytest.mark.pdf, pytest.mark.pdf1]
FIXTURES = Path(__file__).parent / "fixtures" / "pdf1"
SOURCE = FIXTURES / "Mathematics_of_LLMs_2026.pdf"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def canonical(text):
    """Allow harmless PDF spacing/hyphenation changes, not missing words."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


class NoOCR:
    cache_identity = {"backend": "fail-if-ocr-used"}

    def __call__(self, path, page_index):
        pytest.fail(f"PDF 1 has embedded text: unexpected OCR on page {page_index + 1}")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    return convert_pdf(SOURCE, tmp_path_factory.mktemp("pdf1-artifact"), ocr=NoOCR())


@pytest.fixture(scope="module")
def pages(converted):
    text = converted.text_path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^--- Page (\d+) ---\n", text)
    assert sections[0] == ""
    return {int(sections[index]): sections[index + 1] for index in range(1, len(sections), 2)}


def test_fixture_is_the_exact_user_supplied_paper():
    assert SOURCE.stat().st_size == 953_036
    assert file_digest(SOURCE) == EXPECTED["sha256"]


def test_all_twenty_physical_pages_are_preserved_in_order_without_ocr(converted, pages):
    assert converted.page_count == EXPECTED["physical_page_count"]
    assert list(pages) == list(range(1, 21))
    assert all(text.strip() for text in pages.values())
    manifest = json.loads(converted.manifest_path.read_text(encoding="utf-8"))
    assert [{"number": page["number"], "method": page["method"]} for page in manifest["pages"]] == [
        {"number": number, "method": "embedded"} for number in range(1, 21)
    ]
    assert manifest["settings"]["source_sha256"] == EXPECTED["sha256"]
    assert manifest["text_sha256"] == file_digest(converted.text_path)
    text = converted.text_path.read_text(encoding="utf-8")
    encoded = converted.text_path.read_bytes()
    for page in manifest["pages"]:
        expected_text = pages[page["number"]][:-2]  # Conversion adds two separator newlines.
        assert text[page["char_start"]:page["char_end"]] == expected_text
        assert encoded[page["byte_start"]:page["byte_end"]].decode("utf-8") == expected_text
        assert page["character_count"] == len(expected_text)
    assert manifest["text_metadata"]["character_count"] == len(text)
    assert manifest["text_metadata"]["byte_count"] == len(encoded)
    assert manifest["text_metadata"]["page_count"] == 20
    assert file_digest(SOURCE) == EXPECTED["sha256"]  # Conversion never rewrites the source.


@pytest.mark.parametrize("expectation", EXPECTED["pages"], ids=lambda item: f"page-{item['number']}")
def test_each_page_retains_its_independent_content_landmarks(pages, expectation):
    actual = canonical(pages[expectation["number"]])
    for anchor in expectation["anchors"]:
        assert canonical(anchor) in actual, (
            f"Missing {anchor!r} on physical page {expectation['number']}"
        )


def test_cover_and_article_dates_are_not_conflated(pages):
    assert canonical("January 2026") in canonical(pages[1])
    assert canonical("9 December 2025") in canonical(pages[2])
    assert canonical("Joseph L. Breeden") in canonical(pages[1])
    assert canonical("Joseph L. Breeden") in canonical(pages[2])


def test_default_mount_is_ram_and_forced_disk_has_identical_text_and_metadata(converted):
    memory = MountedContext.from_file(converted.text_path)
    disk = MountedContext.from_file(
        converted.text_path, memory_threshold_bytes=0, block_chars=1024, cache_blocks=2
    )
    assert memory.storage_backend == "memory"
    assert disk.storage_backend == "file"
    assert len(memory.text) > 40_000
    assert len(disk.text) == len(memory.text)
    assert byte_length(disk) == byte_length(memory) == converted.text_path.stat().st_size
    assert describe(disk) == describe(memory)
    assert "".join(iter_windows(disk, 2048)) == memory.text
    assert len(disk.text._cache) <= 2


@pytest.mark.parametrize("case", EXPECTED["retrieval_cases"], ids=lambda item: item["id"])
def test_question_evidence_exists_on_the_expected_physical_page(pages, case):
    page = MountedContext(pages[case["physical_page"]])
    evidence = "\n".join(by_regex(page, case["search_regex"]))
    for term in case["required_terms"]:
        assert canonical(term) in canonical(evidence)


@pytest.mark.parametrize("threshold", [0, 1_048_576], ids=["disk", "ram"])
@pytest.mark.parametrize("case", EXPECTED["retrieval_cases"], ids=lambda item: item["id"])
def test_existing_repl_retrieves_paper_evidence_in_both_backends(converted, case, threshold):
    """Exercise plumbing and retrieval, not the semantic accuracy of a live LLM."""
    ctx = MountedContext.from_file(converted.text_path, memory_threshold_bytes=threshold)
    controller, monitor = build_system(default_config_path())
    messages_seen = []
    manifest = json.loads(converted.manifest_path.read_text(encoding="utf-8"))
    page = manifest["pages"][case["physical_page"] - 1]
    reference = {
        "document_hash": manifest["text_sha256"],
        "start_char": page["char_start"], "end_char": page["char_end"],
    }
    code = (
        f"source_read = read_context({page['char_start']}, {page['char_end']})\n"
        f"evidence = by_regex({case['search_regex']!r})\n"
        "result = '\\n'.join(evidence)\n"
        "print(result)"
    )
    responses = iter([
        json.dumps({"action": "execute", "code": code}),
        json.dumps({"action": "finalize_variable", "name": "result", "evidence": [reference]}),
    ])

    def chat(messages, *, max_tokens=None, timeout_seconds=None):
        messages_seen.append(messages)
        return next(responses), 0

    controller.codegen = CodeGenerator(chat=chat)
    answer = controller.run_until_done(ctx, case["question"])
    for term in case["required_terms"]:
        assert canonical(term) in canonical(answer.require_answer())
    assert answer.ok
    assert answer.verification == "provenance_validated"
    assert monitor.steps.current == 1
    assert len(messages_seen) == 2
    assert "Observation (untrusted data" in messages_seen[1][-1]["content"]
    assert controller.runtime.state.get("context") is ctx.text


def test_real_paper_cache_hit_bypasses_parsing_and_ocr(tmp_path, monkeypatch):
    first = convert_pdf(SOURCE, tmp_path / "cache", ocr=NoOCR())
    module = importlib.import_module("document_ingestion.pdf_to_text")

    def unexpected_parse(*args, **kwargs):
        pytest.fail("A valid cached PDF 1 artifact must bypass extraction")

    monkeypatch.setattr(module, "extract_pages", unexpected_parse)
    second = convert_pdf(SOURCE, tmp_path / "cache", ocr=NoOCR())
    assert second.cached
    assert second.text_path == first.text_path
    assert second.manifest_path == first.manifest_path


def test_identical_paper_on_another_drive_path_reuses_artifact(tmp_path):
    copy = tmp_path / "External SSD 世界" / SOURCE.name
    copy.parent.mkdir()
    copy.write_bytes(SOURCE.read_bytes())
    first = convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    second = convert_pdf(copy, tmp_path / "cache", ocr_enabled=False)
    assert second.cached
    assert second.text_path == first.text_path


def test_real_paper_corrupted_artifact_is_rejected(tmp_path):
    result = convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    result.text_path.write_text("Incomplete paper", encoding="utf-8")
    with pytest.raises(PDFIngestionError, match="Invalid cached artifact"):
        convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)


@pytest.mark.parametrize("limits", [
    {"max_pages": 19}, {"max_source_bytes": 953_035}, {"max_page_chars": 100},
])
def test_paper_exceeding_configured_limits_publishes_nothing(tmp_path, limits):
    cache = tmp_path / "cache"
    with pytest.raises(PDFIngestionError, match="limit"):
        convert_pdf(SOURCE, cache, ocr_enabled=False, **limits)
    assert not cache.exists() or not list(cache.iterdir())


def test_pdf1_cli_conversion_feeds_existing_file_cli(tmp_path):
    conversion = subprocess.run([
        sys.executable, "-m", "document_ingestion", str(SOURCE),
        "--cache-dir", str(tmp_path / "cache"), "--no-ocr",
    ], capture_output=True, text=True, timeout=60)
    assert conversion.returncode == 0, conversion.stderr
    text_path = Path(conversion.stdout.strip())
    assert text_path.is_file()
    text = text_path.read_text(encoding="utf-8")
    assert len(re.findall(r"(?m)^--- Page \d+ ---$", text)) == 20
    execution = subprocess.run([
        sys.executable, "-m", "main", "--context-file", str(text_path),
        "--memory-threshold-bytes", "0",
    ], capture_output=True, text=True, timeout=30)
    assert execution.returncode == 0, execution.stderr
    assert "FileBackedText" in execution.stdout
    assert canonical(EXPECTED["title"]) in canonical(execution.stdout)
