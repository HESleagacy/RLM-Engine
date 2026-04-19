"""Small helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("config root must be a mapping")
    return data


def count_paragraphs(text: str) -> int:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return len(blocks)


def has_code_fence(text: str) -> bool:
    return "```" in text
