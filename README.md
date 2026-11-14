# RLM Engine — Recursive Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-242%20passing-brightgreen.svg)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A controlled reasoning engine that thinks under control, under cost, and under verification.**

Implementation of the **Recursive Language Model** architecture from *Khattab et al., 2025*. Instead of dumping entire documents into an LLM's context window, the RLM generates executable Python code in a REPL loop to programmatically explore, chunk, and reason over arbitrarily long contexts.

---

## Quick Start

### Docker (macOS, Windows, Linux)

Install Docker Desktop on macOS/Windows (using Linux containers), or Docker Engine
with the Compose plugin on Linux. Docker supplies Python and all dependencies;
no host Python installation is required. The image supports ARM64 and AMD64.

Create `.env` from `.env.example` and set `GROQ_API_KEY` for live LLM calls.
Compose reads this file automatically; it is never copied into the image.
The homepage and document generator work without a key.

```bash
docker compose up --build -d
docker compose logs -f app
```

Open **http://localhost:7860**. The container uses Gunicorn rather than Flask's
development server and runs as a non-root user. The port is exposed only on the
local device.

Run the CLI without starting the web server:

```bash
docker compose run --rm --no-deps app rlm --prompt "Hello world"
docker compose run --rm --no-deps app rlm --use-groq --prompt "The sky is blue." --query "What color is the sky?"
```

Run offline tests in a separate image (no API key required):

```bash
docker build --target test -t rlm-engine-test .
docker run --rm --init rlm-engine-test
```

Stop the demo:

```bash
docker compose down
```

Dependencies are installed at image build time; live inference and remote
benchmark downloads still require internet access. This setup uses the existing
Linux `fork` execution backend, including on Windows hosts. It does not make
generated Python safe for untrusted public use: the engine still lacks
container-per-execution isolation and memory limits. Do not expose it publicly
or mount sensitive host directories into the container.

### File-backed contexts (internal drive or external SSD)

Use `--context-file` to read an existing UTF-8 document without first loading it
as a command-line string:

```bash
rlm --context-file "/path/to/document.txt" --use-groq --query "Summarize the key findings."
```

Files up to **1 MiB** are kept in RAM. Larger files stay on disk, with a streaming
indexing pass to record character offsets and a bounded, per-worker LRU cache
(8 blocks of 65,536 characters by default). File contents and line endings are
preserved exactly; unlike `--prompt`, file inputs are not stripped or normalized.
The source must remain present and unchanged while a run is in progress.

Options:

- `--memory-threshold-bytes`: adjust the RAM cutoff; `0` forces non-empty files
  to use disk-backed access.
- `--context-block-chars`: characters per cached block.
- `--context-cache-blocks`: maximum cached blocks; `0` disables caching.
- `--context-cache-bytes`: estimated DRAM budget for cached blocks and their
  bookkeeping (4 MiB by default). Oversized blocks are read but not retained.

In Python:

```python
from layer1_input import MountedContext

ctx = MountedContext.from_file("/path/to/document.txt")
print(ctx.storage_backend)  # "memory" or "file"
print(ctx.text[:500])       # Read only the requested characters
```

Disk-backed REPL contexts support `len(context)` and indexing/slicing, not the
full string API. `peek_head`, `peek_tail`, `by_keyword`, and `by_regex` work with
both backends. Use `for chunk in iter_chunks(65536): ...` to process chunks
lazily. `chunker()` remains a list-returning compatibility tool; it collects
all chunks in RAM, as do unrestricted full slices and large search result lists.
File indexing uses memory proportional to the number of blocks, not document
characters; individual reads and line searches still need room for the requested
slice or longest line. Storage backing is not a hard memory limit.

For Docker, set `RLM_DATA_DIR` in `.env` to an existing host document directory.
It can point to an internal drive or an external SSD; Docker Desktop must have
access to that directory. The optional Compose overlay mounts it at `/data`
read-only:

```bash
docker compose -f compose.yaml -f compose.storage.yaml run --rm app rlm --context-file /data/document.txt --use-groq --query "Summarize the key findings."
```

Documents in the repository's `data/` directory are excluded from image builds.
Keep other private documents outside the build context. Mount only the directory
you intend the engine to read, not an entire drive.

### Local indexing, metadata, and DRAM acceleration

The existing sparse character index enables slicing. An additional persistent
**SQLite FTS5 index** enables ranked keyword retrieval over overlapping chunks:

```bash
# Local retrieval; no LLM/API key required. --search implies --index.
rlm --context-file document.txt --search "rotary position embedding"

# Inspect statistics and structural previews without inference.
rlm --context-file document.txt --index --show-metadata

# Make indexed search available to the reasoning controller.
rlm --context-file document.txt --index --use-groq --query "Explain RoPE."
```

The default index directory is `~/.cache/rlm-engine/indexes`; `--index-dir` can
point to an internal drive or external SSD. Source SHA-256, chunk settings, and
index schema/normalization version determine reuse. Source files are not changed
and do not need to be writable. Index directories contain document text: keep
them private. Index creation scans the input; reusing an index still hashes the
source to validate identity. There is no automatic on-disk index eviction.

Results contain a ranked score, bounded snippet, original character span, line
range, and physical page-marker range where available. Read the original span
to inspect the source. Queries use literal-token AND matching, not embeddings
or semantic reasoning. FTS operators are not executed. Up to 64 terms/1,000
query characters and 100 results are supported. Chunks default to 2,048
characters with 128-character overlap (`--index-chunk-chars`, `--index-overlap`).
Character chunking can split unusually long tokens; full line filtering remains
available. PDF spacing such as `isRotary` is split in the **index only**; returned
snippets can therefore have normalized spacing while spans address exact input.

```python
from layer1_input import MountedContext, describe
from layer5_context_access import ContextIndex

ctx = MountedContext.from_file("document.txt")
index = ContextIndex.build(ctx, "/path/to/private-index-cache")
print(index.search("rotary embedding", limit=5))
print(index.cache_info)
print(describe(ctx))
```

Metadata now includes exact byte/character/whitespace-word counts, LF-based line
counts, paragraph counts, code-fence detection, page-marker counts, and heading
hints with character offsets. Heading/page previews are capped at 32 entries;
the search index retains all detected page markers. Headings are heuristic
(Markdown/numbered titles), not a semantic outline. `estimated_tokens` is a rough
four-characters-per-token estimate, not a tokenizer count or billing measure.

**DRAM is used in three places:** small inputs stay resident, file reads use an
LRU block cache, and repeated searches use a bounded hot-result LRU. The search
cache budget defaults to 4 MiB (`--index-cache-bytes`), split between estimated
hot-result storage and SQLite's requested page-cache budget. Hot entries include
query-key/bookkeeping estimates and are also capped at 1,024 entries. A zero
budget disables these caches. The OS may additionally cache disk pages.

Python caches are per process/worker; they are not a shared cross-worker RAM
pool. SQLite connections are opened read-only for queries and are not shared
across forks. Cache budgets are not hard process-RAM limits: returned objects,
index construction, Python allocators, and OS caching consume additional memory.

### Local Python installation

#### 1. Install

```bash
# Core install
pip install -e .

# With Groq LLM support (recommended)
pip install -e ".[groq]"

# Dev dependencies
pip install -e ".[dev]"
```

#### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_...
```

#### 3. Run

```bash
# Full RLM reasoning loop from a source checkout
PYTHONPATH=src python3 -m src.main \
  --use-groq \
  --prompt "The sky is blue. The grass is green. The sun is yellow. The ocean is deep blue." \
  --query "What color is the sky and the ocean?"

# Single-round mode (no LLM required)
PYTHONPATH=src python3 -m src.main --prompt "Hello world"

# Installed CLI
rlm --prompt "Hello world"
```

---

## PDF input (preprocessing, not an engine change)

Convert a PDF to a UTF-8 text artifact, then pass that artifact to the existing
context loader. No PDF-specific controller logic or REPL tools are involved.

```bash
pip install -e ".[pdf]"
rlm-pdf "/path/to/report.pdf" --no-ocr
```

The command prints **only the completed text file's absolute path**. Use that
path with the normal CLI:

```bash
rlm --context-file "/printed/path/content.txt" --use-groq --query "Summarize the report."
```

### Scanned or mixed PDFs

```bash
pip install -e ".[ocr]"
rlm-pdf "/path/to/scanned-report.pdf"
```

OCR additionally requires the **Tesseract executable** and language data on the
host. Install these through your OS package manager or use the Docker image below.
OCR runs locally; PDFium renders pages and Tesseract recognizes text. Conversion
itself makes no cloud requests.

- Embedded text is used whenever a page contains at least one non-whitespace
  character. Pages with content but insufficient text are OCR'd individually.
- For pages containing only a short caption over a scanned image, raise
  `--min-text-chars` to trigger OCR (a heuristic, not layout classification).
- `--language eng+hin`, `--dpi 150`, and `--ocr-timeout 60` configure OCR.
  The selected Tesseract language packs must be installed.
- Truly blank pages without a content stream get a page marker without OCR.
  OCR that returns no readable text is an error, not silent data loss.
- `--no-ocr` explicitly fails when a page requires OCR.

### Artifacts and limits

Pages are written incrementally, with `--- Page N ---` separators. The default
cache is `~/.cache/rlm-engine/pdf`; use `--cache-dir` to put it on an internal drive
or external SSD. Each content-addressed artifact contains:

```text
<artifact-id>/
├── content.txt
└── manifest.json   # Provenance, statistics, page spans/methods, text checksum
```

The manifest records source filename/size/hash, available embedded PDF metadata
(without guessing missing author/title fields), UTC conversion time, aggregate
text statistics, and each page's extraction method and exact UTF-8 byte and
character offsets. Page spans exclude separators. Counts and offsets are
validated against the real 20-page acceptance fixture. Metadata schema changes
invalidate previous conversion-cache keys; old artifacts are not modified.

Conversions are reused only when the source hash and conversion settings match,
and cached text is checksum-verified. Failed conversions leave no completed
artifact. Encrypted, malformed, and zero-page PDFs fail before reasoning starts.
Conversion defaults limit inputs to 100 MiB and 1,000 pages, extracted text to
1,000,000 characters per page, and OCR renders to 25 million pixels. The Python
API accepts overrides; `--max-pages` is also available in the CLI.

The resulting text uses the same 1 MiB RAM/disk cutoff as ordinary text files.
Tables, columns, diagrams, and reading order may not survive plain-text
conversion accurately; page markers are not verified citations. OCR quality
depends on scan quality and language data. Parsing/rendering is not a security
sandbox, and these validation limits are not hard process memory limits.

Artifacts contain document information. Keep the cache private and remove it
explicitly when no longer needed. Conversion caches do not expire automatically.

### Docker with local OCR

The default web image stays lightweight. The optional PDF image includes
Tesseract with English language data, PDFium, and the PDF Python dependencies:

```bash
docker build --target pdf -t rlm-engine-pdf .
docker run --rm --init --mount "type=bind,src=/absolute/document-directory,dst=/data,readonly" --mount "type=volume,src=rlm-pdf-cache,dst=/home/rlm/.cache" rlm-engine-pdf rlm-pdf /data/report.pdf
```

Use the printed container path with `rlm` in the same image and cache volume:

```bash
docker run --rm --init --mount "type=volume,src=rlm-pdf-cache,dst=/home/rlm/.cache,readonly" rlm-engine-pdf rlm --context-file "/home/rlm/.cache/rlm-engine/pdf/<artifact-id>/content.txt"
```

Add `--env-file .env` to `docker run`, then `--use-groq --query "..."` to `rlm`
for live inference. Live inference may send document excerpts to the LLM
provider; PDF preprocessing alone is local.

---

## Resource failure reporting

Recoverable resource failures are reported with stable error codes:

| Code | Meaning |
| --- | --- |
| `disk_full` | Disk space exhausted (`ENOSPC`) or storage quota reached (`EDQUOT`) |
| `out_of_memory` | Python allocation failure (`MemoryError`) or OS `ENOMEM` |
| `worker_terminated` | An execution/OCR worker died; the cause is unknown, not confirmed OOM |

The PDF and engine CLIs print an actionable error to stderr and exit with status
1. Python conversion/mount APIs raise `shared.resource_errors.ResourceExhaustedError`
with a `.code`; worker death raises `WorkerTerminatedError`. `RuntimeEngine.execute()`
returns `ok=False`, `error`, and `error_code`. The multi-round controller stops and
passes the exception to its caller instead of returning an empty/successful answer
or making further LLM calls.

The demo API returns `{"error": "...", "code": "..."}` with HTTP 503 for resource
exhaustion and HTTP 502 for unexpected worker termination. Interrupted conversion
attempts clean up owned temporary artifacts when the process can unwind; they do
not publish partial documents.

This is **error reporting, not resource prevention**: no disk quotas, automatic
on-disk cache eviction, or hard process-RAM limits are imposed. DRAM cache budgets
do not bound all process allocations. A main process killed by the OS cannot
return an error or run cleanup. Reporting under severe memory exhaustion is
best-effort.

---

## Demo: Consistency Challenge

A **Flask + vanilla JS** UI for running the RLM pipeline on fact-consistency tasks.

**The problem:** Given a document with 15 facts scattered across filler paragraphs, write a story using all facts correctly. The RLM controller mounts the complete document and reasons over it through the REPL.

```bash
# Install demo dependencies
pip install -e ".[demo]"

# Set your Groq API key
export GROQ_API_KEY=your_key

# Run the UI
PYTHONPATH=src python demo/app.py
```

Then open **http://localhost:7860** and:

1. Click **Generate Document** to create a fact-scattered document
2. Click **Run RLM** to process the document through the controller REPL
3. Inspect the generated story and fact-accuracy score

> **Stack:** Flask backend (`demo/app.py`) + vanilla JS/CSS frontend (`demo/static/index.html`) — no build step required.

---

## Architecture

The system is organized into **8 distinct layers**, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────┐
│                    CLI / main.py                        │
├─────────────────────────────────────────────────────────┤
│  L1 Input        │ Mount raw text as immutable P       │
├──────────────────┼──────────────────────────────────────┤
│  L2 Controller   │ REPL loop: plan → codegen → exec    │  ← Hot Path
├──────────────────┼──────────────────────────────────────┤
│  L3 Execution    │ Validated exec worker + stdout       │
├──────────────────┼──────────────────────────────────────┤
│  L4 Recursion    │ llm_query() sub-LLM orchestration   │
├──────────────────┼──────────────────────────────────────┤
│  L5 Context      │ Probe, filter, chunk, traverse       │
├──────────────────┼──────────────────────────────────────┤
│  L6 Output       │ Intermediate store → FINAL() lock    │
├──────────────────┼──────────────────────────────────────┤
│  L7 Control      │ Step limits, budget, recursion guard  │
├──────────────────┼──────────────────────────────────────┤
│  L8 Evaluation   │ Benchmarks, baselines, metrics        │
└─────────────────────────────────────────────────────────┘
```

### How It Works

1. **Mount** — Your prompt/document is mounted as a protected `context` variable in a Python sandbox
2. **Generate** — The root LLM (Llama 3.3-70B) generates Python code to explore the context
3. **Execute** — Validated code runs in a killable worker with stdout capture
4. **Recurse** — Generated code can call `llm_query()` to delegate to a sub-LLM (Llama 3.1-8B)
5. **Iterate** — REPL output feeds back to the LLM for the next round
6. **Finalize** — The LLM emits `FINAL(answer)` to lock the output

### Tools Available in the REPL

| Tool | Description |
|------|-------------|
| `llm_query(prompt)` | Query a sub-LLM (~200K char context) |
| `peek_head(n)` | Read first `n` chars of context |
| `peek_tail(n)` | Read last `n` chars of context |
| `by_keyword(*kw)` | Filter lines by keywords |
| `by_regex(pattern)` | Filter lines by regex |
| `chunker(size, overlap)` | Collect fixed windows in a list |
| `iter_chunks(size, overlap=0)` | Stream fixed windows on demand |
| `context_info()` | Document counts and bounded structural previews |
| `search_context(query, limit=5)` | Ranked local keyword matches when `--index` is enabled |
| `print()` | Observe intermediate results |

---

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml):

| Setting | Default | Description |
|---------|---------|-------------|
| `max_steps` | 100 | Hard cap on execution steps |
| `max_recursion_depth` | 5 | Max depth for nested `llm_query()` calls |
| `token_budget` | 100,000 | Total token budget across all LLM calls |
| `max_rounds` | 20 | Max REPL iterations before forced finalization |
| `stdout_truncation` | 3,000 | Chars of `print()` output fed back per round |
| `root_model` | `llama-3.3-70b-versatile` | Root controller LLM |
| `sub_model` | `llama-3.1-8b-instant` | Sub-call LLM for `llm_query()` |
| `sandbox_strict` | `true` | Restrict imports, introspection, and dangerous builtins |

---

## Testing

```bash
# Run all tests
pytest -q

# Run the execution/storage workflow tests
pytest -q tests/test_storage_workflow.py

# Run only the RAM/disk storage workflow regressions
pytest -q -m storage

# Run PDF preprocessing tests
pytest -q -m pdf

# Run acceptance tests against the supplied Mathematics of LLMs paper
pytest -q -m pdf1

# Run simulated disk-full, memory-exhaustion, and worker-death tests
pytest -q -m resources

# Run persistent indexing, metadata, and DRAM cache tests
pytest -q -m indexing

# Verbose output
pytest -v
```

**Current local status (Tesseract and OCR extras available): 242 passing across 13 test suites.**
The real-OCR test requires Tesseract and the OCR extras and is mandatory in the
`pdf-test` Docker target.

Tests run offline with mocked LLM responses and temporary document files.
See [`tests/README.md`](tests/README.md) for the storage contract and CI coverage.

| Test Suite | Coverage |
|------------|----------|
| `test_input_layer.py` | Layer 1: normalize, mount, metadata |
| `test_file_context.py` | File context API, UTF-8 slicing, metadata, serialization, and validation |
| `test_storage_workflow.py` | Bounded I/O, caching, source lifecycle, multi-round REPL, and CLI errors |
| `test_pdf_ingestion.py` | Scanned/mixed PDF edge cases, OCR limits/errors, and malformed inputs |
| `test_pdf1_acceptance.py` | 49 real-paper acceptance cases: extraction, page landmarks, caching, retrieval, RAM/disk, and CLI |
| `test_code_generator.py` | Layer 2: LLM output → executable Python |
| `test_recursion.py` | Layer 4: sub-LLM invocation via guard |
| `test_output.py` | Layer 6: intermediate → FINAL aggregation |
| `test_control.py` | Layer 7: step limits, budget, recursion guard |
| `test_evaluation.py` | Layer 8: metrics + benchmark smoke test |
| `test_regressions.py` | Cross-layer security and control regressions |
| `test_resource_errors.py` | Resource classification, partial-file cleanup, worker death, CLI exit codes, and API responses |
| `test_indexing.py` | Persistent search, UTF-8/page spans, richer metadata, cache budgets/hits, invalidation, and CLI/REPL integration |

---

## Evaluation & Benchmarks

Layer 8 includes benchmark loaders and baseline agents for systematic evaluation:

### Benchmarks

| Benchmark | Complexity | Source |
|-----------|-----------|--------|
| **S-NIAH** | Constant | Synthetic needle-in-haystack (2^13 → 2^18 chars) |
| **BrowseComp** | Multi-hop | Tevatron/browsecomp-plus (1K documents) |
| **OOLONG** | Linear | oolongbench/oolong-synth (trec_coarse) |
| **OOLONG-Pairs** | Quadratic | Pair-aggregation over OOLONG contexts |
| **CodeQA** | Variable | THUDM/LongBench-v2 code repository understanding |

### Baseline Agent

| Agent | Strategy |
|-------|----------|
| **Summary Agent** | Iterative chunk summarization |

### Metrics

- `exact_match()` / `f1_token_overlap()` — accuracy
- `total_cost()` — token + step cost tracking
- `approx_complexity()` — scaling behavior classification

---

## Project Structure

```
RLM-Engine/
├── configs/default.yaml          # Runtime configuration
├── src/
│   ├── main.py                   # CLI entry point + system wiring
│   ├── layer1_input/             # Raw loader, MountedContext, metadata
│   ├── layer2_controller/        # RootController, CodeGenerator, Planner
│   ├── layer3_execution/         # RuntimeEngine, sandbox, StateStore
│   ├── layer4_recursion/         # RecursionManager, sub-LLM invoker
│   ├── layer5_context_access/    # Probe, filter, chunker, traversal
│   ├── layer6_output/            # OutputManager, FINAL() finalizer
│   ├── layer7_control/           # Budget, steps, recursion guards
│   ├── layer8_evaluation/        # Benchmarks, baselines, metrics
│   └── shared/                   # Types, constants, Groq client, utils
├── tests/                        # Per-layer and PDF ingestion regression suites
├── pyproject.toml                # Build config + dependencies
└── CONTEXT.md                    # Detailed architecture analysis
```

---

## References

- Khattab et al., *Recursive Language Models*, 2025
- System prompt adapted from Appendix D.1 of the paper

---

*Built as a research playground — not a production system. Contributions welcome.*
