# RLM Engine — Recursive Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen.svg)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A controlled reasoning engine that thinks under control, under cost, and under verification.**

Implementation of the **Recursive Language Model** architecture from *Khattab et al., 2025*. Instead of dumping entire documents into an LLM's context window, the RLM generates executable Python code in a REPL loop to programmatically explore, chunk, and reason over arbitrarily long contexts.

---

## Quick Start

### 1. Install

```bash
# Core install
pip install -e .

# With Groq LLM support (recommended)
pip install -e ".[groq]"

# Dev dependencies
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_...
```

### 3. Run

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
| `chunker(size, overlap)` | Split context into fixed windows |
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

# Run a specific layer's tests
pytest -q tests/test_execution.py

# Verbose output
pytest -v
```

**Current status: 19 tests passing across 10 test suites.**

| Test Suite | Coverage |
|------------|----------|
| `test_input_layer.py` | Layer 1: normalize, mount, metadata |
| `test_controller.py` | Layer 2: single-round controller flow |
| `test_code_generator.py` | Layer 2: LLM output → executable Python |
| `test_execution.py` | Layer 3: sandboxed exec + result capture |
| `test_recursion.py` | Layer 4: sub-LLM invocation via guard |
| `test_context_access.py` | Layer 5: probe, filter, chunk |
| `test_output.py` | Layer 6: intermediate → FINAL aggregation |
| `test_control.py` | Layer 7: step limits, budget, recursion guard |
| `test_evaluation.py` | Layer 8: metrics + benchmark smoke test |
| `test_regressions.py` | Cross-layer security and control regressions |

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
├── tests/                        # Per-layer test suites (19 tests passing)
├── pyproject.toml                # Build config + dependencies
└── CONTEXT.md                    # Detailed architecture analysis
```

---

## References

- Khattab et al., *Recursive Language Models*, 2025
- System prompt adapted from Appendix D.1 of the paper

---

*Built as a research playground — not a production system. Contributions welcome.*
