"""Raw Context Loader — normalize arbitrary input before mounting."""

from __future__ import annotations


def normalize(text: str) -> str:
    """Normalize whitespace; input is treated as UTF-8 text."""
    if not isinstance(text, str):
        raise TypeError("prompt must be str")
    return text.replace("\r\n", "\n").strip()
