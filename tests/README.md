# Offline regression tests

Install the project with its development dependencies, then run:

```bash
python -m pytest -q
```

For only the hybrid storage contract:

```bash
python -m pytest -q -m storage
```

For PDF ingestion:

```bash
python -m pytest -q -m pdf
```

For the supplied Mathematics of LLMs paper only:

```bash
python -m pytest -q -m pdf1
```

For resource-failure handling:

```bash
python -m pytest -q -m resources
```

For indexing, richer metadata, and DRAM caches:

```bash
python -m pytest -q -m indexing
```

For Layer 2 controller and provider-accounting regressions:

```bash
python -m pytest -q -m controller
```

No API key, external SSD, internet connection, or downloaded benchmark is needed.
Temporary directories represent either an internal disk or a mounted external
drive. In-process TCP connections are blocked by `conftest.py` so accidental
provider calls fail instead of spending API credits. CLI subprocess tests use
local deterministic execution or fail validation before provider initialization.

## Storage contract

| Contract | Regression coverage |
| --- | --- |
| Default cutoff stays at 1 MiB, inclusive | Below/at/above-boundary and empty-file cases |
| Cutoff uses UTF-8 bytes, not character count | Multibyte text with different byte/character lengths |
| Small files are immutable in-memory snapshots | Original whitespace retained; source can be removed after mounting |
| Large files are not retained as a full string | Instrumented indexing/read sizes; only requested blocks loaded |
| Cache hits avoid disk reads | Actual read counts, LRU eviction/reload, disabled cache |
| Disk slices behave like string slices | UTF-8, emoji, CRLF, negative indices/steps, bounds, invalid steps |
| Chunk iteration remains lazy | No reads before consumption; stopping early leaves remaining chunks unread |
| Overlapping chunks retain all source characters | RAM/disk backends, multiple sizes and overlap values |
| Searches stream lines | Keyword/regex results and read instrumentation |
| Metadata stays consistent across backends | Empty input, line endings, paragraph boundaries, code fences |
| Input cannot silently change mid-run | Replacement while indexing, truncation, deletion, same-size replacement, modification during line traversal |
| No writes to source documents | Read-only files; spaces and Unicode in mount paths |
| Worker transfer does not copy the cache | Serialization and real multi-round worker execution |
| REPL tools keep working with disk contexts | Head probe, keyword/regex search, lazy chunks, legacy chunks, file alias |
| Reusing the controller does not leak documents | Disk → RAM → disk runs with distinct queries |
| Unavailable storage fails safely | Worker errors and preserved mounted context |
| CLI reports actionable failures | Missing file, invalid UTF-8/settings, conflicting input flags |

`test_file_context.py` contains focused API and representation checks.
`test_storage_workflow.py` contains I/O, lifecycle, CLI, and controller workflows.
Superseded cache, source-change, and single-round tool tests were consolidated
into these stronger checks; unrelated per-layer tests remain.
The old `test_execution.py`, `test_controller.py`, and `test_context_access.py`
smoke tests were removed: workflow tests now assert actual state persistence,
values, tool results, and failures rather than just successful execution.
The duplicate text CLI happy path was also removed; the real-paper CLI test
covers it. The mutually-exclusive input flag check remains.

## PDF preprocessing contract

Temporary `context`/`query` rebinding remains an intentional limitation.
`test_regressions.py` checks that it is allowed inside one execution but cannot
replace either mounted value in later rounds, for RAM/disk and inline/worker
execution. No strict rebinding prohibition is implemented or expected by tests.

`test_pdf1_acceptance.py` uses the unchanged, user-provided paper in
`tests/fixtures/pdf1/`. Its 49 acceptance cases check:

- Exact input identity and all 20 physical pages in order, without OCR.
- Independent text landmarks on every page, including front matter and references.
- Distinct cover/article dates and correct physical-versus-printed page numbering.
- Equivalent RAM/disk text and metadata, bounded caching, and streamed reconstruction.
- Six evidence-retrieval cases: author, vocabulary size, Q/K/V terminology,
  RoPE expansion, KL divergence, and the four listed limitations.
- Each retrieval case through the existing controller in both storage backends.
- Strict JSON actions and actual source-read references for both RAM/disk runs,
  returning `provenance_validated` outcomes rather than legacy marker responses.
- Cache reuse, corruption checks, source relocation, limits, and CLI-to-CLI flow.

Expected answers and page landmarks are in `tests/fixtures/pdf1/expected.json`.
They come from the supplied paper, not a snapshot of the parser output. Assertions
normalize whitespace/hyphenation and Unicode compatibility forms; they do not
assert that mathematical typesetting survives extraction.

The controller uses scripted LLM responses to retrieve evidence. These tests
verify ingestion/execution plumbing, **not live LLM reasoning accuracy**.

`test_pdf_ingestion.py` retains synthetic cases only for behaviors the real paper
cannot cover:

- Mixed embedded/scanned/blank pages and UTF-8 OCR output.
- Source/settings-based cache invalidation.
- Atomic publication and cleanup after disabled/empty/failed OCR.
- Source changes during conversion and malformed/encrypted/empty PDFs.
- Missing dependencies, invalid OCR settings, command arguments, subprocess
  timeouts/errors, pixel limits, and renderer/temp-file cleanup.
- Sparse-text OCR routing, actual PDFium rendering, and optional real Tesseract.

Seven older happy-path test functions (10 parameterized cases) were removed
after their contracts were replaced by the real-paper checks. Other storage and
per-layer tests remain because they cover separate engine contracts.

The real OCR test is skipped locally when Tesseract or rendering dependencies
are absent. The `pdf-test` image installs these dependencies and sets
`RLM_REQUIRE_OCR_TEST=1`, making their presence mandatory instead of skipping.

These are correctness tests, not throughput benchmarks. They avoid flaky timing
and resident-memory thresholds: read instrumentation checks bounded access
directly. The cache is not a hard memory limit—full slices, long lines, and large
search results can still allocate large strings/lists.

## Resource-failure contract

`test_resource_errors.py` and resource-marked OCR cases inject failures rather
than exhausting the actual test machine. They cover:

- `ENOSPC`, `EDQUOT`, `ENOMEM`, `MemoryError`, and wrapped causes; unrelated file
  or validation errors must not be mislabeled.
- Failures creating the cache, writing/closing text, saving manifests, and
  publishing artifacts; no incomplete or temporary artifact survives a handled
  failure.
- Parser allocation errors and cache reads: memory failure is not corruption.
- RAM mounting and disk indexing failures, inline and worker execution,
  serialization, parent receive, validation, and worker startup failures.
- A worker exiting or being killed by a signal; errors do not falsely claim a
  known OOM cause. Only a newly created test worker is terminated.
- Controller termination without further LLM calls, nonzero CLI exit codes,
  structured HTTP errors, and releasing request slots after failures.
- OCR image-save failures, renderer cleanup, and unexpected subprocess death.

No disk or process-wide RAM quotas or automatic on-disk eviction are implemented. True
host-wide OOM and killing the main process cannot guarantee a Python response.
The primary controller now reports these failures in a noncompleted `RunOutcome`;
conversion/mount APIs still raise resource exceptions. Callers using
`require_answer()` receive `RunFailedError`, never a successful error string.

## Indexing and DRAM contract

`test_indexing.py` covers:

- Exact size/word counts, bounded heading/page previews, and offsets crossing
  scan-block boundaries, with equivalent RAM/disk metadata.
- Persistent SQLite FTS5 reuse and source/chunk-setting invalidation.
- Ranked local results, original character/line/page ranges, and all page markers
  even when preview lists are capped.
- Safe literal queries, bounded query/result sizes, empty input, corruption,
  missing indexes, and disk/memory error propagation.
- Hot results avoiding SQLite, independent result objects, LRU eviction, disabled
  caches, and accounting for query keys (including empty-result queries).
- File-block byte budgets, oversized uncached reads, and cache hit statistics.
- PDF spacing normalization without source changes; real-paper indexed retrieval
  and the original REPL reading returned spans.
- Exact UTF-8 PDF page spans, source provenance, and offline CLI inspection.

DRAM budgets cover estimated Python-cache storage and requested SQLite page
cache, not all process allocations or the OS file cache. Caches are per process,
not a cross-worker shared-memory service. Token counts are explicitly estimates;
heading detection and page markers are structural hints, not semantic assertions.

## Layer 2 contract

`test_layer2_controller.py` and `test_model_calls.py` cover the structured
controller and its model boundary:

- Exactly one strict JSON action; duplicate/mixed fields, invalid names/types,
  oversized data, marker strings, Unicode, apostrophes, and explicit legacy mode.
- Typed terminal outcomes, finite repair allowance, missing/empty variables,
  partial candidates, round/step limits, repeated actions, and execution failures.
- Host-issued source-read references: wrong hashes, unread/changed spans, bounded
  ledgers, provenance-only labeling, and optional external verifier acceptance.
- Terminal flow behavior, cancellation during requests/execution, busy-controller
  protection, and cleanup after a run. Only test-owned execution workers are killed.
- Bounded head/tail stdout, separate errors, retained latest diagnostics after
  compaction, safe bounded state summaries, and preflight input-window rejection.
- Atomic input/output reservations, concurrent reservation safety, recorded
  overruns, conservative unknown charges, and budgets preserved between runs.
- Bounded retry/backoff/Retry-After behavior, permanent-error classification,
  deadlines, SDK retries disabled, JSON mode, and request timeout forwarding.
- Explicit subcall-counter reset in inline and worker paths, and child-to-parent
  reconciliation of known/uncertain usage.
- CLI failures returning nonzero and HTTP failures never producing/scoring a story.

No live LLM requests are made. Transport tests use synthetic providers and a
fake Groq client, not real credentials. Tokenizer-independent accounting uses a
conservative byte estimate; it is not a guarantee of the provider's billed usage.
Phase/state progress is heuristic; provenance validation is not a truth checker.
The supported primary return type is `RunOutcome`, replacing `FinalAnswer`.

## Continuous integration

`.github/workflows/tests.yml` runs the full suite on Linux and macOS with Python
3.10 and 3.12. A Linux Docker job also runs the test image as its non-root user
and exercises the Compose CLI with a read-only host document mount. Windows
execution is supported through Linux containers, not the native `fork` backend.
An additional Docker job exercises PDF conversion with real local OCR.

To reproduce the container tests:

```bash
docker build --target test -t rlm-engine-test .
docker run --rm --init rlm-engine-test
```

```bash
docker build --target pdf-test -t rlm-pdf-test .
docker run --rm --init rlm-pdf-test
```
