"""Streaming document statistics and bounded structural previews."""

from __future__ import annotations

import re
from typing import Callable

from layer1_input.context_repr import MountedContext
from layer1_input.file_context import FileBackedText
from shared.resource_errors import report_resource_errors
from shared.types import HeadingHint, PageMarker, StructureHints

_PAGE = re.compile(r"--- Page ([1-9]\d*) ---")
_MARKDOWN_HEADING = re.compile(r"(#{1,6})\s+(.+)")
_NUMBERED_HEADING = re.compile(r"(\d{1,3}(?:\.\d{1,3}){0,5})\s+([A-Z][^\n]+)")


@report_resource_errors
def scan_metadata(
    ctx: MountedContext, *, on_page: Callable[[PageMarker], None] | None = None
) -> StructureHints:
    """Scan bounded blocks; optionally stream every page marker to an index."""
    text = ctx.text
    length = len(text)
    newline_count = paragraphs = words = page_count = heading_count = 0
    has_content = newline_seen = in_word = code_fence = inside_code = False
    tail = prefix = ""
    line_start = 0
    pages: list[PageMarker] = []
    headings: list[HeadingHint] = []

    def inspect_line() -> None:
        nonlocal page_count, heading_count, inside_code
        if len(prefix) > 512:
            return
        line = prefix.strip()
        if line.startswith("```"):
            inside_code = not inside_code
            return
        if inside_code:
            return
        page = _PAGE.fullmatch(line)
        if page:
            marker = PageMarker(int(page[1]), line_start)
            page_count += 1
            if len(pages) < 32:
                pages.append(marker)
            if on_page is not None:
                on_page(marker)
            return
        markdown = _MARKDOWN_HEADING.fullmatch(line)
        numbered = _NUMBERED_HEADING.fullmatch(line)
        if markdown:
            heading = HeadingHint(markdown[2].strip(), len(markdown[1]), line_start)
        elif numbered and not re.search(r"\.\s*\.\s*\.", line):
            heading = HeadingHint(line, numbered[1].count(".") + 1, line_start)
        else:
            return
        heading_count += 1
        if len(headings) < 32:
            headings.append(heading)

    for start in range(0, length, 65_536):
        block = text[start:start + 65_536]
        newline_count += block.count("\n")
        code_fence = code_fence or "```" in tail + block
        tail = (tail + block)[-2:]
        for char in block:
            whitespace = char.isspace()
            if not whitespace and not in_word:
                words += 1
            in_word = not whitespace
            if char == "\n":
                if newline_seen and has_content:
                    paragraphs += 1
                    has_content = False
                newline_seen = True
            elif not whitespace:
                has_content = True
                newline_seen = False
        pieces = block.split("\n")
        cursor = start
        for index, piece in enumerate(pieces):
            prefix += piece[:max(0, 513 - len(prefix))]
            if index < len(pieces) - 1:
                inspect_line()
                prefix = ""
                line_start = cursor + len(piece) + 1
            cursor += len(piece) + 1
    inspect_line()
    return StructureHints(
        line_count=newline_count + 1 if length else 0,
        paragraph_count=paragraphs + int(has_content),
        has_code_fence=code_fence,
        character_count=length,
        byte_count=byte_length(ctx),
        word_count=words,
        estimated_tokens=(length + 3) // 4,
        page_count=page_count,
        heading_count=heading_count,
        page_markers=tuple(pages),
        headings=tuple(headings),
    )


def describe(ctx: MountedContext) -> StructureHints:
    return scan_metadata(ctx)


def byte_length(ctx: MountedContext) -> int:
    if isinstance(ctx.text, FileBackedText):
        return ctx.text.byte_length
    return len(ctx.text.encode("utf-8"))
