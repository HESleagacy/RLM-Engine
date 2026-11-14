"""Optional local PDF rendering and Tesseract OCR; no cloud requests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from document_ingestion.text_extractor import PDFIngestionError
from shared.resource_errors import WorkerTerminatedError, report_resource_errors


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


@dataclass(frozen=True)
class TesseractOCR:
    language: str = "eng"
    dpi: int = 150
    timeout_seconds: float = 60
    max_pixels: int = 25_000_000

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", self.language):
            raise ValueError("OCR language must be a Tesseract language code, e.g. eng or eng+hin")
        if not 72 <= self.dpi <= 600:
            raise ValueError("OCR dpi must be between 72 and 600")
        if self.timeout_seconds <= 0 or self.max_pixels <= 0:
            raise ValueError("OCR timeout and pixel limit must be positive")

    @property
    def cache_identity(self) -> dict:
        executable = shutil.which("tesseract")
        binary = Path(executable).stat() if executable else None
        return {
            "backend": "tesseract",
            "language": self.language,
            "dpi": self.dpi,
            "timeout_seconds": self.timeout_seconds,
            "max_pixels": self.max_pixels,
            "pdfium_version": package_version("pypdfium2"),
            "pillow_version": package_version("pillow"),
            # Updating the executable invalidates previously extracted OCR text.
            "executable": executable,
            "executable_stamp": [binary.st_size, binary.st_mtime_ns] if binary else None,
        }

    @report_resource_errors
    def __call__(self, path: Path, page_index: int) -> str:
        executable = shutil.which("tesseract")
        if not executable:
            raise PDFIngestionError(
                'Scanned page requires OCR: install Tesseract and pip install -e ".[ocr]"'
            )
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise PDFIngestionError('Install OCR rendering with pip install -e ".[ocr]"') from exc

        try:
            with tempfile.TemporaryDirectory(prefix="rlm-ocr-") as directory:
                image_path = Path(directory) / "page.png"
                with pdfium.PdfDocument(str(path)) as document:
                    page = document[page_index]
                    try:
                        width, height = page.get_size()
                        scale = self.dpi / 72
                        if (width * scale + 1) * (height * scale + 1) > self.max_pixels:
                            raise PDFIngestionError("PDF page exceeds the OCR pixel limit")
                        bitmap = page.render(scale=scale)
                        try:
                            image = bitmap.to_pil()
                            try:
                                image.save(image_path)
                            finally:
                                image.close()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                result = subprocess.run(
                    [executable, str(image_path), "stdout", "-l", self.language],
                    capture_output=True, text=True, encoding="utf-8",
                    timeout=self.timeout_seconds, check=True,
                )
                return result.stdout
        except PDFIngestionError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise PDFIngestionError("OCR timed out") from exc
        except subprocess.CalledProcessError as exc:
            if exc.returncode < 0:
                raise WorkerTerminatedError(
                    f"OCR worker terminated by signal {-exc.returncode}. "
                    "The cause is unknown; memory exhaustion cannot be confirmed."
                ) from exc
            raise PDFIngestionError(f"Tesseract failed: {exc.stderr.strip()}") from exc
        except Exception as exc:
            raise PDFIngestionError(f"OCR rendering failed: {exc}") from exc
