# Offline regression tests

Install the project with its development dependencies, then run:

```bash
python -m pytest -q
```

For only the hybrid storage contract:

```bash
python -m pytest -q -m storage
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

These are correctness tests, not throughput benchmarks. They avoid flaky timing
and resident-memory thresholds: read instrumentation checks bounded access
directly. The cache is not a hard memory limit—full slices, long lines, and large
search results can still allocate large strings/lists.

## Continuous integration

`.github/workflows/tests.yml` runs the full suite on Linux and macOS with Python
3.10 and 3.12. A Linux Docker job also runs the test image as its non-root user
and exercises the Compose CLI with a read-only host document mount. Windows
execution is supported through Linux containers, not the native `fork` backend.

To reproduce the container tests:

```bash
docker build --target test -t rlm-engine-test .
docker run --rm --init rlm-engine-test
```
