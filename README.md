# Layer extension and testing guide

This document explains how to swap or extend each layer of the Layered Reasoning System (RLM), what contracts you must preserve, and how to verify changes with the existing test suite or new tests.

**Project layout:** Python package roots live under `src/` (`pyproject.toml` sets `pythonpath = ["src"]` for pytest). Imports look like `from layer1_input import ...`, not `from src.layer1_input`.

**Run all tests** (from the repository root, with dev dependencies installed):

```bash
pip install -e ".[dev]"
pytest -q
```

**Run one layer’s tests:**

```bash
pytest -q tests/test_input_layer.py
pytest -q tests/test_controller.py
pytest -q tests/test_code_generator.py
pytest -q tests/test_execution.py
pytest -q tests/test_recursion.py
pytest -q tests/test_context_access.py
pytest -q tests/test_output.py
pytest -q tests/test_control.py
pytest -q tests/test_evaluation.py
```

**Smoke the CLI** after wiring changes in `src/main.py`:

```bash
python -m src.main --prompt "Hello"
python -m src.main --prompt "Hello" --use-groq   # needs GROQ_API_KEY and `pip install -e ".[groq]"`
```

---

## 1. How the system is wired today

Understanding the default path avoids changing the wrong abstraction.

| Layer | Used directly by `build_system()` / `main`? | Notes |
|-------|-----------------------------------------------|--------|
| 1 — Input | Yes | `MountedContext`, `normalize()` |
| 2 — Controller | Yes | `RootController`, `CodeGenerator` |
| 3 — Execution | Yes | `RuntimeEngine`, `StateStore`, `ToolInterface` |
| 4 — Recursion | No | Not imported in `main.py`; use via `RecursionManager` in your own orchestration |
| 5 — Context access | No | Not used inside `RootController.run_round`; useful for custom controllers or preprocessing |
| 6 — Output | Yes (default) | `RootController` constructs `OutputManager()` if you do not inject one |
| 7 — Control | Yes | `StepLimiter`, `BudgetManager`, `RecursionGuard` → `ExecutionMonitor` |
| 8 — Evaluation | No | Library-style metrics and benchmarks |

`RootController.run_round(ctx, instruction)` does the following today:

1. Asks `Planner.next_step(self.flow.iteration, has_more=True)`.
2. If the plan is `PlanAction.STOP`, returns `{"ok": False, "reason": "stop"}`.
3. Otherwise calls `CodeGenerator.generate(instruction, ctx.text[:2000])` (first 2000 characters of mounted text only).
4. Runs the returned code with `RuntimeEngine.execute`.
5. Advances `ControlFlow` and, on success, appends `result` to `OutputManager.intermediate`.

So **Layer 5 is not on the default hot path** until you change the controller to call it. **Layer 4** is similarly **off the default CLI path** until you integrate it.

---

## 2. General pattern for replacing a layer

1. **Identify the contract:** public methods, argument types, return types, and side effects (for example, `RuntimeEngine.execute` must return `ExecutionResult` from `shared.types`).
2. **Prefer injection over globals:** most layers are plain classes constructed in `build_system()` or tests. Subclass or provide an alternative instance and pass it into the parent (`RootController(runtime=..., codegen=...)`, etc.).
3. **Keep `__init__.py` exports stable** if other code imports the package API (`layer1_input`, `layer2_controller`, …). You may add symbols; avoid renaming without updating all imports.
4. **Test in isolation first** (unit test the new class), then **integration test** the smallest stack that exercises your change (often `RootController` + `RuntimeEngine` + `MountedContext`).
5. **Run the full suite** before you consider the change done.

Shared types live in `src/shared/types.py` (`ExecutionResult`, `LLMCallable`, `StructureHints`, etc.).

---

## 3. Layer 1 — Input handling

### Role

Normalize raw user text, represent it as an immutable mounted prompt `P` (`MountedContext`), and expose metadata (length, structure hints).

### Files

| File | Responsibility |
|------|----------------|
| `src/layer1_input/raw_loader.py` | `normalize(text: str) -> str` |
| `src/layer1_input/context_repr.py` | `MountedContext` dataclass (`frozen=True`) |
| `src/layer1_input/metadata_provider.py` | `describe(ctx)`, `byte_length(ctx)` |
| `src/layer1_input/__init__.py` | Re-exports public API |

### Contracts

- `MountedContext` must stay **immutable** after construction (`frozen=True`). New representations should still treat `text` as the single source of truth for `P`.
- `normalize` currently requires a real `str` and normalizes line endings and outer whitespace.

### How to change the implementation

- **New normalization rules:** edit or replace `normalize()` in `raw_loader.py`. Keep the name and signature if you want zero call-site changes, or add a parallel function and update `layer1_input.__init__.py` and importers.
- **Richer context (e.g. attachments, URI):** extend `MountedContext` with additional **frozen** fields, or introduce a new dataclass and update every consumer (`RootController` currently only uses `ctx.text`).
- **Metadata:** extend `StructureHints` in `shared/types.py` if you add new fields, and update `describe()` accordingly.

### How to test

```bash
pytest -q tests/test_input_layer.py
```

**After changes:** add assertions for new normalization or metadata fields. If you extend `MountedContext`, grep for `MountedContext` and update or add tests wherever it is constructed.

---

## 4. Layer 2 — Root controller (planning, codegen, control flow)

### Role

Orchestrate “what to do next,” produce executable Python, and advance iteration state. `CodeGenerator` can call an LLM or use a deterministic fallback.

### Files

| File | Responsibility |
|------|----------------|
| `src/layer2_controller/controller.py` | `RootController` |
| `src/layer2_controller/planner.py` | `Planner`, `PlanAction`, `PlanStep` |
| `src/layer2_controller/code_generator.py` | `CodeGenerator` |
| `src/layer2_controller/control_flow.py` | `ControlFlow` |
| `src/layer2_controller/__init__.py` | Exports `RootController`, `Planner`, `PlanAction` |

### Contracts

- **`CodeGenerator.generate(instruction, context_excerpt) -> str`** must return Python that **`exec()` can run** and that ends with a meaningful **`result`** assignment in normal success cases. The implementation strips fenced markdown, coerces invalid syntax to `result = ...`, and appends `result = None` if the model forgot `result`.
- **`Planner.next_step(step_index, *, has_more) -> PlanStep`:** today `run_round` only treats **`STOP`** specially; other actions all lead to code generation. If you implement real branching (`READ_CHUNK` vs `GENERATE_CODE`), **update `RootController.run_round`** to honor those actions.
- **`ControlFlow`:** `advance()` increments `iteration`; used as input to the planner.

### How to change the implementation

- **New planning policy:** subclass `Planner` or replace `Planner.next_step`, then pass `RootController(..., planner=your_planner)`. If you add new `PlanAction` values, update `controller.py` logic.
- **Different codegen (new model, templates):** subclass `CodeGenerator` or inject `CodeGenerator(make_groq_llm(...))` as in `main.build_system`. For tests, inject a fake `LLMCallable` (see `tests/test_code_generator.py`).
- **Multi-step loops:** extend `RootController` with a new method (for example `run_until_done`) or wrap it; keep `run_round` behavior stable for existing tests unless you intentionally migrate them.

### How to test

```bash
pytest -q tests/test_controller.py
pytest -q tests/test_code_generator.py
```

**Integration idea:** build `RootController(RuntimeEngine(...), codegen=CodeGenerator(fake_llm))` and assert the shape of the returned dict and `execution` payload.

---

## 5. Layer 3 — Execution environment

### Role

Execute generated code, merge globals back into `StateStore`, expose tools, optionally sandbox builtins, and enforce step limits via `StepLimiter.tick()` inside `execute`.

### Files

| File | Responsibility |
|------|----------------|
| `src/layer3_execution/runtime_engine.py` | `RuntimeEngine` |
| `src/layer3_execution/state_store.py` | `StateStore` |
| `src/layer3_execution/tool_interface.py` | `ToolInterface` |
| `src/layer3_execution/sandbox.py` | `safe_builtins()` |
| `src/layer3_execution/__init__.py` | Package exports |

### Contracts

- **`RuntimeEngine.execute(code: str) -> ExecutionResult`:** on success, `ok=True` and `value` is `g.get("result")` after `exec`. On failure, `ok=False` and `error` contains traceback text.
- **State persistence:** every non-private name in the exec globals (except tool callables registered in `ToolInterface`) is written back via `StateStore.set`.
- **Sandbox:** `strict_sandbox=True` replaces `__builtins__` with `safe_builtins()` (no `open`, `eval`, `exec`, etc.).

### How to change the implementation

- **Stronger sandbox:** extend `safe_builtins()` or replace `_build_globals()` in a `RuntimeEngine` subclass.
- **New tools:** `tools.register("name", fn)` before `execute`; generated code can then call `name(...)`.
- **Different execution backend** (restricted subprocess, WASM, etc.): subclass `RuntimeEngine` and keep `execute`’s return type as `ExecutionResult`; adapt persistence semantics to match what Layer 2 expects.

### How to test

```bash
pytest -q tests/test_execution.py
```

Add tests that assert sandbox rejection (for example code using `open`) and that tools appear in the namespace.

---

## 6. Layer 4 — Recursive sub-call system

### Role

Depth-limited sub-LLM calls: build a `Subtask`, invoke an `LLMCallable`, merge many results.

### Files

| File | Responsibility |
|------|----------------|
| `src/layer4_recursion/recursion_manager.py` | `RecursionManager` |
| `src/layer4_recursion/subtask_constructor.py` | `Subtask`, `from_prompt` |
| `src/layer4_recursion/sub_llm_invoker.py` | `invoke` |
| `src/layer4_recursion/result_integrator.py` | `merge_strings` |
| `src/layer4_recursion/__init__.py` | Exports |

### Contracts

- **`RecursionManager.run_subtask(prompt)`** wraps work in `RecursionGuard.enter` / `leave`.
- **`LLMCallable`:** `def __call__(self, prompt: str, *, max_tokens: int | None = None) -> str` (see `shared.types`).

### How to change the implementation

- **Different decomposition:** replace `from_prompt` or post-process inside `RecursionManager.run_subtask`.
- **Caching / deduplication:** wrap the `llm` callable or subclass `RecursionManager`.
- **Integration with Layer 2:** call `RecursionManager` from a custom planner or from generated tools (register a tool that calls back into the manager — mind re-entrancy and depth limits).

### How to test

```bash
pytest -q tests/test_recursion.py
pytest -q tests/test_control.py   # RecursionGuard behavior
```

Use a fake LLM that returns deterministic strings. To test depth errors, use `RecursionGuard(max_depth=...)` and nested `enter()` calls as in `test_control.py`.

---

## 7. Layer 5 — Context access strategy

### Role

Read mounted context in small pieces: probe head/tail, filter by keyword or regex, fixed windows, line tree.

### Files

Under `src/layer5_context_access/`: `probe.py`, `filter.py`, `chunker.py`, `traversal.py`, `__init__.py`.

### Contracts

Functions take `MountedContext` (or derived text) and return strings or sequences; they must not mutate `ctx`.

### How to change the implementation

Add new functions or modules and export them from `__init__.py` if they are part of the public API.

### How to test

```bash
pytest -q tests/test_context_access.py
```

### Wiring into the controller

Today `RootController` passes `ctx.text[:2000]` to codegen. To use Layer 5:

1. Inject a custom `CodeGenerator` that calls `peek_head`, `by_keyword`, etc., or
2. Subclass `RootController` and override how the context excerpt is built.

Then add **controller-level** tests that prove the new excerpt logic.

---

## 8. Layer 6 — Output construction

### Role

Collect intermediate values, aggregate, and finalize with `FINAL()` into a locked `FinalAnswer`.

### Files

| File | Responsibility |
|------|----------------|
| `src/layer6_output/intermediate_store.py` | `IntermediateStore` |
| `src/layer6_output/aggregator.py` | `join_text` |
| `src/layer6_output/finalizer.py` | `FINAL`, `FinalAnswer` |
| `src/layer6_output/output_manager.py` | `OutputManager` |
| `src/layer6_output/__init__.py` | Exports |

### Contracts

- `RootController` appends `result` to `output.intermediate` after each successful round.
- `finalize_joined()` returns `FinalAnswer` with `locked=True`.

### How to change the implementation

- **Different aggregation:** subclass `OutputManager` or replace `join_text` usage in `finalize_joined`.
- **Structured outputs:** store dicts in `IntermediateStore` and teach `finalize_joined` to serialize them.

### How to test

```bash
pytest -q tests/test_output.py
```

If you change `RootController`’s interaction with `OutputManager`, update `tests/test_controller.py` or add a dedicated integration test.

---

## 9. Layer 7 — Cost and execution control

### Role

Step limits, token budget, recursion depth tracking, token event accounting, aggregated `ExecutionMonitor.snapshot()`.

### Files

Under `src/layer7_control/`: `step_limiter.py`, `budget_manager.py`, `recursion_guard.py`, `token_tracker.py`, `execution_monitor.py`, `__init__.py`.

### Contracts

- **`StepLimiter.tick()`** is invoked inside `RuntimeEngine.execute` when a limiter is attached; it raises `RuntimeError` when exceeded.
- **`RecursionGuard`** is used by `RecursionManager` for sub-calls (separate from controller’s `ControlFlow` iteration).

### How to change the implementation

- **Tighter budgets:** lower defaults in `configs/default.yaml` under `control:` or pass stricter instances in `build_system()`.
- **Wall clock:** `default.yaml` includes `wall_clock_seconds`, but `ExecutionMonitor` does not enforce it yet; implementing enforcement means extending `ExecutionMonitor` or `RuntimeEngine` and adding tests.

### How to test

```bash
pytest -q tests/test_control.py
pytest -q tests/test_execution.py   # step limiter attached to RuntimeEngine
```

---

## 10. Layer 8 — Evaluation framework

### Role

Benchmark task types, metrics (`exact_match`, cost tuples), and `evaluate_one` helper.

### Files

| Area | Path |
|------|------|
| Evaluator | `src/layer8_evaluation/evaluator.py` |
| Metrics | `src/layer8_evaluation/metrics/` |
| Benchmarks | `src/layer8_evaluation/benchmarks/` |

### Contracts

- **`evaluate_one(gold, predict, *, tokens, steps)`** calls `predict()`, compares with `exact_match`, returns `EvalResult(correct=..., cost=(tokens, steps))`.

### How to change the implementation

- Add benchmark builders in `benchmarks/` and export from `benchmarks/__init__.py`.
- Add metrics in `metrics/` and compose them in `evaluator.py` or a new runner.

### How to test

```bash
pytest -q tests/test_evaluation.py
```

For new benchmarks, add small deterministic examples (like `trivial_example()` / `sample_task()`) and assert gold labels and predictors.

---

## 11. Configuration (`configs/default.yaml`)

Sections map conceptually to layers:

| YAML key | Layer | Consumed in code today? |
|----------|--------|-------------------------|
| `control.*` | 7 | Yes in `build_system()` (`max_steps`, `token_budget`, `max_recursion_depth`) |
| `execution.sandbox_strict` | 3 | Yes |
| `execution.allowed_imports` | 3 | Reserved / not wired in `RuntimeEngine` yet |
| `recursion.max_subcalls_per_step` | 4 | Not read in `main.py` yet |
| `evaluation.benchmarks` | 8 | Not read in `main.py` yet |

When you wire new config keys, **load them in `build_system()`** (or a dedicated factory) and add a test that the YAML value affects behavior.

---

## 12. Checklist after modifying any layer

1. `pytest -q` passes.
2. If you changed public names or signatures, grep the repo and update imports/tests.
3. If behavior crosses layers, add at least one test that constructs the real stack (`MountedContext` → controller → runtime → output).
4. If you rely on LLMs, keep **deterministic unit tests** with fake `LLMCallable` instances; reserve live API calls for manual or optional integration runs.

---

## 13. Quick reference — test file to layer mapping

| Layer | Primary test module |
|-------|---------------------|
| 1 | `tests/test_input_layer.py` |
| 2 | `tests/test_controller.py`, `tests/test_code_generator.py` |
| 3 | `tests/test_execution.py` |
| 4 | `tests/test_recursion.py` |
| 5 | `tests/test_context_access.py` |
| 6 | `tests/test_output.py` |
| 7 | `tests/test_control.py` (+ execution tests when StepLimiter is attached) |
| 8 | `tests/test_evaluation.py` |

This should give you a repeatable loop: **change one layer → run its focused tests → run full `pytest` → optionally smoke `python -m src.main`.**
