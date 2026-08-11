# Context Document

## Project: Recursive Language Model (RLM) Engine

> A modular, 8-layer Python implementation of the RLM architecture (Khattab et al., 2025).
> Functions as a **controlled reasoning engine** where an LLM generates executable Python code
> in a REPL loop to programmatically explore long-context documents and answer queries via
> inference-time scaling.

---

# 1.  INPUT HANDLING LAYER

### Purpose

Ingest arbitrary-length prompts and store them in an external execution environment.

### Subcomponents

* `raw_loader.normalize()` → strip/normalize whitespace from raw text
* `context_repr.MountedContext` → frozen dataclass wrapping the immutable prompt `P`
* `metadata_provider.describe()` → returns `StructureHints` (line count, paragraph count, code fence detection)

### Principle

Input is *mounted*, not passed — the context is stored in the execution environment as a variable.

---

# 2.  ROOT CONTROLLER (Hot Path)

### Purpose

Orchestrates reasoning by generating executable logic in a multi-round REPL loop.

### Subcomponents

* `controller.RootController`
  - `run_round()` — legacy single-round: plan → codegen → exec → record
  - `run_until_done()` — **primary RLM entry**: multi-round REPL loop with FINAL() termination
* `code_generator.CodeGenerator`
  - `generate()` — single-turn fallback (backward-compatible)
  - `generate_step()` — multi-turn chat using the paper's Appendix D.1 system prompt
  - `_parse_step()` — parses `FINAL()`, `FINAL_VAR()`, and ` ```repl` blocks from LLM output
* `planner.Planner` — minimal stub (READ_CHUNK → GENERATE_CODE → STOP)
* `control_flow.ControlFlow` — iteration counter with max_iterations guard

### Weakness

LLMs are poor at long-horizon planning. Planner is currently a stub.

---

# 3.  EXECUTION ENVIRONMENT

### Purpose

Executes generated code and maintains state across rounds.

### Subcomponents

* `runtime_engine.RuntimeEngine` — Python `exec()` sandbox with:
  - stdout capture via `contextlib.redirect_stdout`
  - State persistence across rounds
  - Restricted builtins injection
* `state_store.StateStore` — mutable key-value store surviving across exec rounds
* `tool_interface.ToolInterface` — registry of callable tools injected into exec namespace
* `sandbox.safe_builtins()` — allowlists harmless builtins and rejects imports/dunder introspection

### Principle

Executes, does not decide.

---

# 4.  RECURSIVE SUB-CALL SYSTEM

### Purpose

Decompose problems via sub-LLM calls (`llm_query()` tool).

### Subcomponents

* `recursion_manager.RecursionManager`
  - `run_subtask()` — single sub-call with depth guard enter/leave
  - `run_many()` — parallel sub-calls via `ThreadPoolExecutor(max_workers=5)`
* `subtask_constructor.Subtask` — frozen dataclass (prompt + depth)
* `sub_llm_invoker.invoke()` — thin wrapper calling the LLM
* `result_integrator.merge_strings()` — joins sub-results with separator

### Limitation

Unreliable decomposition, high variance.

### Future

Replace with structured execution (tree/graph).

---

# 5.  CONTEXT ACCESS STRATEGY

### Purpose

Control how data is read from the mounted context.

### Strategies

* `probe.peek_head()` / `peek_tail()` — small reads from context boundaries
* `filter.by_keyword()` / `by_regex()` — line-level filtering
* `chunker.fixed_windows()` — character-level chunking with configurable overlap
* `traversal.lines_as_tree()` — degenerate tree representation (placeholder for real AST)

### Insight

Read less, extract more.

### Tools Registered in REPL

`peek_head`, `peek_tail`, `by_keyword`, `by_regex`, `chunker` (via `main.py` wiring)

---

# 6.  OUTPUT CONSTRUCTION SYSTEM

### Purpose

Build final answer from computed state.

### Subcomponents

* `output_manager.OutputManager` — orchestrates intermediate → aggregate → FINAL
* `intermediate_store.IntermediateStore` — append-only list of partial results
* `aggregator.join_text()` — concatenates parts with separator
* `finalizer.FINAL()` → `FinalAnswer` — immutable locked answer

### Constraint

No regeneration after computation. `FinalAnswer` is frozen.

---

# 7.  COST & EXECUTION CONTROL

### Purpose

Bound system behavior and cost.

### Subcomponents

* `step_limiter.StepLimiter` — hard cap on execution steps (default: 100)
* `budget_manager.BudgetManager` — thread-safe token budget with `spend()`/`remaining()` (default: 100K)
* `recursion_guard.RecursionGuard` — depth limiter with enter/leave (default: 5)
* `token_tracker.TokenTracker` — event-level token recording
* `execution_monitor.ExecutionMonitor` — aggregate health snapshot of all control components

### Principle

Reasoning must be constrained.

---

# 8.  EVALUATION FRAMEWORK

### Benchmarks

* **S-NIAH** → synthetic needle-in-haystack (2^13 to 2^18 chars), constant complexity
* **BrowseComp** → multi-hop reasoning (HuggingFace streaming, 1K documents)
* **OOLONG** → linear scaling (trec_coarse)
* **OOLONG-Pairs** → quadratic scaling
* **CodeQA** → LongBench-v2 code repository understanding

### Baselines

* `SummaryAgent` — iterative chunk summarization agent

### Metrics

* `exact_match()` / `f1_token_overlap()` — Accuracy / F1
* `total_cost()` — Cost (tokens, steps)
* `approx_complexity()` — Scaling behavior

### Principle

If it cannot be measured, it cannot be improved.

---

# DATA FLOW (RLM REPL Loop)

```
User provides --prompt (context) + --query
    │
    ▼
[main.py] build_system() wires all layers
    │
    ▼
[Layer 1] normalize() → MountedContext
    │
    ▼
[Layer 2] RootController.run_until_done()
    │   ├── Mounts ctx.text as `context` in StateStore
    │   └── Loop (max 20 rounds):
    │       ├── CodeGenerator.generate_step() → StepResult
    │       │   ├── System prompt from paper's Appendix D.1
    │       │   ├── Multi-turn chat history (code + REPL output)
    │       │   └── Parses FINAL() / FINAL_VAR() / ```repl blocks
    │       ├── If FINAL → return FinalAnswer
    │       ├── RuntimeEngine.execute(code) (Layer 3)
    │       │   ├── exec() with safe_builtins + tools namespace
    │       │   ├── stdout captured
    │       │   └── State persisted back to StateStore
    │       └── History updated with {code, output}
    │
    ▼
[Layer 6] FinalAnswer (locked, immutable)
```

**Tools available inside REPL:** `llm_query()` (Layer 4), `peek_head()`, `peek_tail()`, `by_keyword()`, `by_regex()`, `chunker()` (Layer 5), plus `print()` for observation.

---

# CONFIGURATION (configs/default.yaml)

| Setting | Default | Layer |
|---------|---------|-------|
| `max_steps` | 100 | L7 |
| `max_recursion_depth` | 5 | L7 |
| `token_budget` | 100,000 | L7 |
| `max_rounds` | 20 | L2 |
| `stdout_truncation` | 3,000 chars | L2 |
| `root_model` | llama-3.3-70b-versatile | L2 |
| `sub_model` | llama-3.1-8b-instant | L4 |
| `sandbox_strict` | true | L3 |

---

# TEST STATUS

**19 tests pass across 10 test modules**:

| Test | Status |
|------|--------|
| `test_code_generator.py` (2 tests) | ✅ |
| `test_context_access.py` | ✅ |
| `test_control.py` | ✅ |
| `test_controller.py` | ✅ |
| `test_evaluation.py` | ✅ |
| `test_execution.py` | ✅ |
| `test_input_layer.py` | ✅ |
| `test_output.py` | ✅ |
| `test_recursion.py` | ✅ |
| `test_regressions.py` | ✅ |

---

# SYSTEM-LEVEL INSIGHT

## What This System Is

A:

* controlled reasoning engine
* executable cognitive pipeline
* inspectable LLM system

## What It Is NOT

* a chatbot
* a wrapper over APIs
* a "vibe-based agent"

---

# CRITICAL DESIGN AXES

### 1. Control vs Flexibility

More control → stability
More flexibility → chaos

### 2. Cost vs Accuracy

Higher accuracy often increases cost
Optimization is mandatory

### 3. Structure vs Exploration

Structure (trees) beats blind recursion

---

# KNOWN ISSUES & PENDING WORK

### Resolved

* ✅ **`_extract_repl_block()` regex** — was using double-escaped `\\n` / `\\s\\S` preventing match of actual newlines in LLM output. Fixed to use proper regex metacharacters.
* ✅ **Evaluation module** — added missing `evaluate_one()` to `evaluator.py`, and `sample_task()` / `trivial_example()` to `benchmarks/__init__.py`.

### Open

* ✅ **`TokenTracker` wiring** — root and recursive LLM token usage is recorded and exposed by `ExecutionMonitor`
* ✅ **Sandbox escape mitigation** — strict execution now validates imports/dunder access and uses a killable worker when a timeout is configured; this is still not a container-level security boundary
* ⚠️ **Planner is a stub** — minimal policy (READ_CHUNK → GENERATE_CODE → STOP), no learned planning

### Future Evolution Path

1. Replace recursion → execution tree
2. Add verification layer
3. Introduce learned planning
4. Optimize cost-aware reasoning
5. Move toward hybrid symbolic + LLM system

---

# KNOWN GLOBAL FAILURE MODES

* Infinite loops
* Recursive explosion
* Context misreading
* Output hallucination
* Cost unpredictability

---

# GLOBAL MITIGATION STRATEGY

* Hard constraints (Layer 7)
* Deterministic execution (Layer 3)
* Structured access (Layer 5)
* Output locking (Layer 6)
* Evaluation (Layer 8)

---

# PROJECT STRUCTURE

```
project-root/
│
├── CONTEXT.md
├── README.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── layer1_input/
│   │   ├── __init__.py
│   │   ├── raw_loader.py
│   │   ├── context_repr.py
│   │   └── metadata_provider.py
│   │
│   ├── layer2_controller/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── planner.py
│   │   ├── code_generator.py
│   │   └── control_flow.py
│   │
│   ├── layer3_execution/
│   │   ├── __init__.py
│   │   ├── runtime_engine.py
│   │   ├── state_store.py
│   │   ├── tool_interface.py
│   │   └── sandbox.py
│   │
│   ├── layer4_recursion/
│   │   ├── __init__.py
│   │   ├── recursion_manager.py
│   │   ├── subtask_constructor.py
│   │   ├── sub_llm_invoker.py
│   │   └── result_integrator.py
│   │
│   ├── layer5_context_access/
│   │   ├── __init__.py
│   │   ├── probe.py
│   │   ├── filter.py
│   │   ├── chunker.py
│   │   └── traversal.py
│   │
│   ├── layer6_output/
│   │   ├── __init__.py
│   │   ├── output_manager.py
│   │   ├── intermediate_store.py
│   │   ├── aggregator.py
│   │   └── finalizer.py
│   │
│   ├── layer7_control/
│   │   ├── __init__.py
│   │   ├── budget_manager.py
│   │   ├── step_limiter.py
│   │   ├── recursion_guard.py
│   │   ├── token_tracker.py
│   │   └── execution_monitor.py
│   │
│   ├── layer8_evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── baselines/
│   │   │   ├── __init__.py
│   │   │   ├── codeact_agent.py
│   │   │   └── summary_agent.py
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   ├── s_niah.py
│   │   │   ├── browsecomp.py
│   │   │   ├── oolong.py
│   │   │   ├── oolong_pairs.py
│   │   │   └── codeqa.py
│   │   └── metrics/
│   │       ├── __init__.py
│   │       ├── accuracy.py
│   │       ├── cost.py
│   │       └── scaling.py
│   │
│   └── shared/
│       ├── __init__.py
│       ├── constants.py
│       ├── types.py
│       ├── utils.py
│       └── groq_client.py
│
└── tests/
    ├── __init__.py
    ├── test_input_layer.py
    ├── test_controller.py
    ├── test_code_generator.py
    ├── test_execution.py
    ├── test_recursion.py
    ├── test_context_access.py
    ├── test_output.py
    ├── test_control.py
    └── test_evaluation.py
```

---

# FINAL PRINCIPLE

> The system must not only think — it must think **under control, under cost, and under verification**

---

End of Document
