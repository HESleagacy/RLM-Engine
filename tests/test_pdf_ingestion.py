"""Targeted PDF/OCR edge cases; real-document happy paths live in PDF 1 tests."""

import json
import errno
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject, DictionaryObject, NameObject, NumberObject,
)

from document_ingestion import convert_pdf
from document_ingestion.ocr_adapter import TesseractOCR
from document_ingestion.text_extractor import PDFIngestionError
from shared.resource_errors import ResourceExhaustedError, WorkerTerminatedError

pytestmark = pytest.mark.pdf


def make_pdf(path, pages):
    """None = blank page, bytes = scanned page, string = embedded text."""
    writer = PdfWriter()
    for content in pages:
        page = writer.add_blank_page(width=600, height=800)
        if content is None:
            continue
        if isinstance(content, bytes):
            image = DecodedStreamObject()
            image.set_data(content)
            image.update({
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(4),
                NameObject("/Height"): NumberObject(4),
                NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                NameObject("/BitsPerComponent"): NumberObject(8),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/XObject"): DictionaryObject({
                    NameObject("/Scan"): writer._add_object(image),
                }),
            })
            commands = b"q 200 0 0 200 0 0 cm /Scan Do Q"
        else:
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({
                    NameObject("/F1"): writer._add_object(font),
                }),
            })
            escaped = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands = f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode("ascii")
        stream = DecodedStreamObject()
        stream.set_data(commands)
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)
    return path


class StubOCR:
    def __init__(self, texts, identity="test-ocr-v1"):
        self.texts = texts
        self.identity = identity
        self.calls = []

    @property
    def cache_identity(self):
        return {"backend": self.identity}

    def __call__(self, path, page_index):
        self.calls.append(page_index)
        return self.texts[page_index]


def test_mixed_pdf_uses_ocr_only_for_scanned_pages_and_writes_utf8(tmp_path):
    source = make_pdf(tmp_path / "mixed.pdf", ["Embedded", b"\xff" * 16, None])
    ocr = StubOCR({1: "OCR text: 世界 🙂"})
    result = convert_pdf(source, tmp_path / "cache", ocr=ocr)
    assert ocr.calls == [1]
    assert "OCR text: 世界 🙂" in result.text_path.read_text(encoding="utf-8")
    manifest = json.loads(result.manifest_path.read_text())
    assert [page["method"] for page in manifest["pages"]] == ["embedded", "ocr", "blank"]
    assert result.page_count == 3


def test_source_content_and_ocr_settings_invalidate_cache(tmp_path):
    source = make_pdf(tmp_path / "input.pdf", ["Original"])
    root = tmp_path / "cache"
    first = convert_pdf(source, root, ocr=StubOCR({}, "v1"))
    settings_change = convert_pdf(source, root, ocr=StubOCR({}, "v2"))
    assert first.text_path != settings_change.text_path
    make_pdf(source, ["Replacement"])
    source_change = convert_pdf(source, root, ocr=StubOCR({}, "v1"))
    assert source_change.text_path != first.text_path
    assert "Replacement" in source_change.text_path.read_text()


@pytest.mark.parametrize("failure", ["disabled", "empty", "exception"])
def test_failed_conversion_publishes_no_partial_artifacts(tmp_path, failure):
    source = make_pdf(tmp_path / "mixed.pdf", ["First succeeds", b"\xff" * 16])
    cache = tmp_path / "cache"
    ocr = StubOCR({1: ""})
    if failure == "exception":
        def unavailable(*args):
            raise PDFIngestionError("OCR unavailable")
        ocr = type("FailedOCR", (), {
            "cache_identity": {"backend": "failed"},
            "__call__": unavailable,
        })()
    with pytest.raises(PDFIngestionError):
        convert_pdf(source, cache, ocr=ocr, ocr_enabled=failure != "disabled")
    assert not list(cache.iterdir())


def test_modified_source_during_conversion_is_rejected(tmp_path):
    source = make_pdf(tmp_path / "scan.pdf", [b"\xff" * 16])

    class ModifyingOCR(StubOCR):
        def __call__(self, path, page_index):
            make_pdf(path, ["Changed while OCR was running"])
            return "old text"

    with pytest.raises(PDFIngestionError, match="changed during conversion"):
        convert_pdf(source, tmp_path / "cache", ocr=ModifyingOCR({}))
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("kind", ["malformed", "encrypted", "empty"])
def test_invalid_pdfs_are_rejected(tmp_path, kind):
    source = tmp_path / "invalid.pdf"
    if kind == "malformed":
        source.write_bytes(b"not a PDF")
    else:
        writer = PdfWriter()
        if kind == "encrypted":
            writer.add_blank_page(width=100, height=100)
            writer.encrypt("secret")
        with source.open("wb") as output:
            writer.write(output)
    with pytest.raises(PDFIngestionError):
        convert_pdf(source, tmp_path / "cache", ocr_enabled=False)
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("settings", [
    {"min_text_chars": 0}, {"max_pages": 0},
    {"max_source_bytes": -1}, {"max_page_chars": 0},
])
def test_invalid_conversion_settings_are_rejected(tmp_path, settings):
    source = make_pdf(tmp_path / "input.pdf", ["Text"])
    with pytest.raises(ValueError):
        convert_pdf(source, tmp_path / "cache", **settings)


def test_missing_pdf_dependency_has_actionable_error(tmp_path, monkeypatch):
    source = make_pdf(tmp_path / "input.pdf", ["Text"])
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(PDFIngestionError, match="Install PDF support"):
        convert_pdf(source, tmp_path / "cache", ocr_enabled=False)


def test_missing_tesseract_has_actionable_error(tmp_path, monkeypatch):
    source = make_pdf(tmp_path / "scan.pdf", [b"\xff" * 16])
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(PDFIngestionError, match="install Tesseract"):
        convert_pdf(source, tmp_path / "cache")


@pytest.fixture
def fake_ocr_renderer(monkeypatch):
    events = []

    class Image:
        def save(self, path):
            events.append("save")
            path.write_bytes(b"rendered image")

        def close(self):
            events.append("image-close")

    class Bitmap:
        def to_pil(self):
            return Image()

        def close(self):
            events.append("bitmap-close")

    class Page:
        def get_size(self):
            return 600, 800

        def render(self, *, scale):
            events.append(("render", scale))
            return Bitmap()

        def close(self):
            events.append("page-close")

    class Document:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("document-close")

        def __getitem__(self, index):
            events.append(("page", index))
            return Page()

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=Document))
    monkeypatch.setattr(shutil, "which", lambda name: sys.executable)
    return events


def test_ocr_runs_local_command_with_timeout_and_cleans_up(tmp_path, monkeypatch, fake_ocr_renderer):
    image_paths = []

    def run(command, **kwargs):
        assert command[2:] == ["stdout", "-l", "eng+hin"]
        assert kwargs["timeout"] == 12
        assert kwargs["check"] is True
        assert kwargs["encoding"] == "utf-8"
        image_paths.append(Path(command[1]))
        assert image_paths[0].read_bytes() == b"rendered image"
        return SimpleNamespace(stdout="Recognized text")

    monkeypatch.setattr(subprocess, "run", run)
    assert TesseractOCR(language="eng+hin", dpi=144, timeout_seconds=12)(
        tmp_path / "input.pdf", 2
    ) == "Recognized text"
    assert ("page", 2) in fake_ocr_renderer
    assert ("render", 2) in fake_ocr_renderer
    assert all(event in fake_ocr_renderer for event in [
        "image-close", "bitmap-close", "page-close", "document-close",
    ])
    assert not image_paths[0].exists()


@pytest.mark.parametrize("kind", ["timeout", "process"])
def test_ocr_process_failures_are_actionable(tmp_path, monkeypatch, fake_ocr_renderer, kind):
    def run(command, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        raise subprocess.CalledProcessError(1, command, stderr="language data missing")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(
        PDFIngestionError, match="timed out" if kind == "timeout" else "language data missing"
    ):
        TesseractOCR()(tmp_path / "input.pdf", 0)
    assert "document-close" in fake_ocr_renderer


def test_oversized_ocr_page_is_rejected_before_rendering(tmp_path, fake_ocr_renderer):
    with pytest.raises(PDFIngestionError, match="pixel limit"):
        TesseractOCR(max_pixels=10)(tmp_path / "input.pdf", 0)
    assert not any(isinstance(event, tuple) and event[0] == "render" for event in fake_ocr_renderer)
    assert "page-close" in fake_ocr_renderer
    assert "document-close" in fake_ocr_renderer


@pytest.mark.resources
@pytest.mark.parametrize(("failure", "code"), [
    (MemoryError(), "out_of_memory"),
    (OSError(errno.ENOSPC, "simulated full OCR temp storage"), "disk_full"),
])
def test_ocr_image_save_resource_failure_closes_renderer(tmp_path, monkeypatch, fake_ocr_renderer, failure, code):
    original_write = Path.write_bytes

    def write(path, data):
        if path.name == "page.png":
            raise failure
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(ResourceExhaustedError) as caught:
        TesseractOCR()(tmp_path / "input.pdf", 0)
    assert caught.value.code == code
    assert all(event in fake_ocr_renderer for event in [
        "image-close", "bitmap-close", "page-close", "document-close",
    ])


@pytest.mark.resources
def test_ocr_process_killed_by_signal_is_not_mislabeled_as_oom(tmp_path, monkeypatch, fake_ocr_renderer):
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(-9, command, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(WorkerTerminatedError, match="signal 9") as caught:
        TesseractOCR()(tmp_path / "input.pdf", 0)
    assert "cannot be confirmed" in str(caught.value)


def test_sparse_embedded_text_can_be_explicitly_routed_to_ocr(tmp_path):
    source = make_pdf(tmp_path / "sparse.pdf", ["x"])
    ocr = StubOCR({0: "Full page recovered"})
    result = convert_pdf(source, tmp_path / "cache", ocr=ocr, min_text_chars=10)
    assert ocr.calls == [0]
    assert "Full page recovered" in result.text_path.read_text()


@pytest.mark.parametrize("settings", [
    {"language": "../eng"}, {"dpi": 0},
    {"timeout_seconds": 0}, {"max_pixels": 0},
])
def test_invalid_ocr_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        TesseractOCR(**settings)


def test_real_pdfium_rendering_with_only_recognition_mocked(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    Image = pytest.importorskip("PIL.Image")
    source = make_pdf(tmp_path / "scan.pdf", [b"\xff" * 16])
    monkeypatch.setattr(shutil, "which", lambda name: sys.executable)

    def recognize(command, **kwargs):
        with Image.open(command[1]) as rendered:
            assert rendered.size == (600, 800)
            rendered.verify()
        return SimpleNamespace(stdout="Text from rendered page")

    monkeypatch.setattr(subprocess, "run", recognize)
    assert TesseractOCR(dpi=72)(source, 0) == "Text from rendered page"


def test_real_local_ocr_when_runtime_dependencies_are_available(tmp_path):
    required = os.environ.get("RLM_REQUIRE_OCR_TEST") == "1"
    if required:
        assert shutil.which("tesseract"), "PDF test image must provide Tesseract"
        import pypdfium2
        import PIL.Image
    elif shutil.which("tesseract") is None:
        pytest.skip("Tesseract is not installed; exercised in the PDF Docker test target")
    pytest.importorskip("pypdfium2")
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    ImageFont = pytest.importorskip("PIL.ImageFont")
    image = Image.new("RGB", (900, 200), "white")
    try:
        ImageDraw.Draw(image).text(
            (40, 60), "HELLO OCR", fill="black", font=ImageFont.load_default(size=64)
        )
        source = tmp_path / "real-scan.pdf"
        image.save(source, "PDF", resolution=144)
    finally:
        image.close()
    result = convert_pdf(source, tmp_path / "cache", ocr=TesseractOCR(dpi=144))
    assert "HELLO OCR" in result.text_path.read_text().upper()
