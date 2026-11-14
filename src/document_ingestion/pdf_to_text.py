"""Convert PDFs into cached UTF-8 files, independently of the RLM engine."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Protocol

from document_ingestion.artifact_store import ArtifactStore, file_digest, source_stamp
from document_ingestion.ocr_adapter import TesseractOCR, package_version
from document_ingestion.text_extractor import PDFIngestionError, extract_pages
from layer1_input import MountedContext, describe
from shared.resource_errors import (
    ResourceExhaustedError, WorkerTerminatedError, report_resource_errors, resource_error,
)


class OCRBackend(Protocol):
    @property
    def cache_identity(self) -> dict: ...

    def __call__(self, path: Path, page_index: int) -> str: ...


@dataclass(frozen=True)
class ConversionResult:
    text_path: Path
    manifest_path: Path
    page_count: int
    cached: bool


@report_resource_errors
def convert_pdf(
    source: str | Path,
    cache_dir: str | Path,
    *,
    ocr_enabled: bool = True,
    ocr: OCRBackend | None = None,
    min_text_chars: int = 1,
    max_pages: int = 1000,
    max_source_bytes: int = 104_857_600,
    max_page_chars: int = 1_000_000,
) -> ConversionResult:
    """Extract one page at a time and publish only a complete, valid text file."""
    if min_text_chars <= 0 or max_pages <= 0 or max_source_bytes <= 0 or max_page_chars <= 0:
        raise ValueError("Text threshold, page count, file size, and page text limits must be positive")
    source = Path(source).expanduser().resolve(strict=True)
    before = source_stamp(source)
    if before[2] > max_source_bytes:
        raise PDFIngestionError(f"PDF exceeds the {max_source_bytes}-byte size limit")
    ocr = ocr if ocr is not None else TesseractOCR()
    settings = {
        "schema": 2,
        "source_sha256": file_digest(source),
        "extractor": "pypdf",
        "extractor_version": package_version("pypdf"),
        "ocr": ocr.cache_identity if ocr_enabled else None,
        "min_text_chars": min_text_chars,
        "max_pages": max_pages,
        "max_source_bytes": max_source_bytes,
        "max_page_chars": max_page_chars,
    }
    if source_stamp(source) != before:
        raise PDFIngestionError("PDF changed while hashing")
    store = ArtifactStore(cache_dir)
    key = store.key(settings)
    manifest = store.completed(key)
    cached = manifest is not None
    if manifest is None:
        with tempfile.TemporaryDirectory(prefix=".converting-", dir=store.root) as work:
            temporary = Path(work)
            pages = []
            pdf_metadata = {}
            char_cursor = byte_cursor = 0
            text_path = temporary / "content.txt"
            with text_path.open("w", encoding="utf-8", newline="\n") as output:
                for page in extract_pages(source, max_pages=max_pages, document_info=pdf_metadata):
                    text = page.text
                    method = "embedded"
                    if len(text.strip()) < min_text_chars and page.has_contents:
                        if not ocr_enabled:
                            raise PDFIngestionError(
                                f"Page {page.number} requires OCR, but OCR is disabled"
                            )
                        text = ocr(source, page.number - 1)
                        method = "ocr"
                        if not text.strip():
                            raise PDFIngestionError(f"OCR returned no readable text for page {page.number}")
                    elif not text.strip():
                        method = "blank"
                    if len(text) > max_page_chars:
                        raise PDFIngestionError(f"Page {page.number} exceeds the text size limit")
                    marker = f"--- Page {page.number} ---\n"
                    char_start = char_cursor + len(marker)
                    byte_start = byte_cursor + len(marker.encode("utf-8"))
                    char_end = char_start + len(text)
                    byte_end = byte_start + len(text.encode("utf-8"))
                    output.write(marker)
                    output.write(text)
                    output.write("\n\n")
                    pages.append({
                        "number": page.number, "method": method,
                        "char_start": char_start, "char_end": char_end,
                        "byte_start": byte_start, "byte_end": byte_end,
                        "character_count": len(text), "byte_count": byte_end - byte_start,
                        "word_count": sum(1 for _ in re.finditer(r"\S+", text)),
                    })
                    char_cursor, byte_cursor = char_end + 2, byte_end + 2
            if not pages:
                raise PDFIngestionError("PDF has no pages")
            if source_stamp(source) != before:
                raise PDFIngestionError("PDF changed during conversion")
            manifest = {
                "artifact_key": key, "settings": settings, "page_count": len(pages),
                "pages": pages, "text_sha256": file_digest(text_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "name": source.name, "byte_count": before[2],
                    "sha256": settings["source_sha256"], "pdf_metadata": pdf_metadata,
                },
                "text_metadata": asdict(describe(MountedContext.from_file(
                    text_path, memory_threshold_bytes=0
                ))),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            manifest = store.publish(temporary, key)
    return ConversionResult(
        store.root / key / "content.txt", store.root / key / "manifest.json",
        manifest["page_count"], cached,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF to UTF-8 text for the RLM engine")
    parser.add_argument("source", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "rlm-engine" / "pdf")
    parser.add_argument("--no-ocr", action="store_true", help="Fail on scanned pages instead of running OCR")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ocr-timeout", type=float, default=60)
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=1000)
    args = parser.parse_args()
    try:
        result = convert_pdf(
            args.source, args.cache_dir, ocr_enabled=not args.no_ocr,
            ocr=TesseractOCR(args.language, args.dpi, args.ocr_timeout),
            min_text_chars=args.min_text_chars, max_pages=args.max_pages,
        )
    except (ResourceExhaustedError, WorkerTerminatedError) as exc:
        parser.exit(1, f"rlm-pdf: error: [{exc.code}] {exc}\n")
    except (OSError, PDFIngestionError, ValueError, MemoryError) as exc:
        failure = resource_error(exc)
        if failure is not None:
            parser.exit(1, f"rlm-pdf: error: [{failure.code}] {failure}\n")
        parser.error(str(exc))
    # stdout is just the path, so scripts can pass it to rlm --context-file.
    print(result.text_path)
