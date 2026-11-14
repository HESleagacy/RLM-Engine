# PDF 1: real-document acceptance fixture

User-provided file: `Mathematics_of_LLMs_2026.pdf`.

- Title: *The Simple Mathematics of Large Language Models*
- Author: Joseph L. Breeden
- Cover date: January 2026
- Article date on physical page 2: 9 December 2025
- Physical PDF pages: 20 (not 19)
- SHA-256: `6444d94d202b56eab9158a8b55b0a6da6619836bb847c98f4a3ba223db26ae92`

The paper is retained unchanged as test input. The project's software license
does not relicense the paper; attribution and notices in the PDF remain intact.

`expected.json` contains small, independently selected text landmarks and
retrieval expectations based on the supplied paper, not a golden dump of the
extractor's output. Page numbers in this file are **physical PDF page numbers**.
For example, the RoPE discussion is on physical page 14, printed page 13.

All 20 pages contain embedded text, including the cover. This fixture tests real
PDF text extraction, not OCR accuracy. Synthetic scan and failure fixtures remain
necessary for OCR and malformed/encrypted documents.

Whitespace, line-break hyphenation, and Unicode compatibility forms are ignored
when checking text landmarks. Mathematical layout and equation typesetting are
not expected to round-trip through plain text.
