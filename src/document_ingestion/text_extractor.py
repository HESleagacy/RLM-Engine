"""Page-at-a-time embedded text extraction using the optional pypdf backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class PDFIngestionError(ValueError):
    """Conversion failed; no completed artifact should be published."""


@dataclass(frozen=True)
class ExtractedPage:
    number: int
    text: str
    has_contents: bool


def extract_pages(path: Path, *, max_pages: int, document_info: dict | None = None) -> Iterator[ExtractedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFIngestionError('Install PDF support with pip install -e ".[pdf]"') from exc

    try:
        with path.open("rb") as source:
            reader = PdfReader(source)
            if reader.is_encrypted:
                raise PDFIngestionError("Encrypted PDFs are not supported; decrypt the file first")
            if document_info is not None:
                metadata = getattr(reader, "metadata", None) or {}
                for name, key in (
                    ("title", "/Title"), ("author", "/Author"), ("subject", "/Subject"),
                    ("creator", "/Creator"), ("producer", "/Producer"),
                    ("creation_date", "/CreationDate"), ("modification_date", "/ModDate"),
                ):
                    value = metadata.get(key)
                    if value is not None:
                        document_info[name] = str(value)[:2048]
            count = len(reader.pages)
            if not count:
                raise PDFIngestionError("PDF has no pages")
            if count > max_pages:
                raise PDFIngestionError(f"PDF exceeds the {max_pages}-page limit")
            for index in range(count):
                page = reader.pages[index]
                yield ExtractedPage(
                    number=index + 1,
                    text=page.extract_text() or "",
                    has_contents=page.get_contents() is not None,
                )
    except PDFIngestionError:
        raise
    except Exception as exc:
        raise PDFIngestionError(f"Could not extract PDF text: {exc}") from exc
