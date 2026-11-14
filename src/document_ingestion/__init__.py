"""Optional document preprocessing; the reasoning engine remains text-only."""

from document_ingestion.pdf_to_text import ConversionResult, convert_pdf

__all__ = ["ConversionResult", "convert_pdf"]
