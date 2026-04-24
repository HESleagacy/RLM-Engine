# Recursive Language Model (RLM) Playground

This project is a **playground and implementation guide** for building and testing Recursive Language Models (RLMs) as described in the paper *Recursive Language Models* (Khattab et al., 2025). 

It is designed as a modular, layered Python system where you can swap out planning strategies, execution environments, or LLM backends to experiment with inference-time scaling and long-context handling.

## 🚀 Getting Started

### 1. Installation
The project uses `pyproject.toml`. For the full RLM experience (including Groq-powered LLMs), install with the `groq` extra:

```bash
pip install -e ".[groq]"
```

*Note: If you have an older version of setuptools, a `setup.py` shim is provided to enable editable installs.*

### 2. Environment Setup
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_key_here
```

### 3. Run the RLM Playground
You can run the full recursive loop directly from the CLI. This mounts your prompt as a "long document" (`context`) and allows the LLM to programmatically explore it to answer your query.

```bash
PYTHONPATH=src python3 -m src.main \
  --use-groq \
  --prompt "The sky is blue. The grass is green. The sun is yellow. The ocean is deep blue. Mountains are tall." \
  --query "What color is the sky and the ocean?"
```

## 🏗️ Architecture: The 8-Layer Pipeline

The system is organized into 8 distinct layers, making it easy to test isolated components of the RLM reasoning chain:

1. **Layer 1: Input** (Raw loader, immutable prompt `P`)
2. **Layer 2: Controller** (**Hot Path**: Multi-round REPL loop, Planner, CodeGen)
3. **Layer 3: Execution** (Python `exec()` sandbox with stdout capture)
4. **Layer 4: Recursion** (`llm_query` tool for sub-LLM orchestration)
5. **Layer 5: Context Access** (Probing, filtering, and chunking tools)
6. **Layer 6: Output** (Intermediate store, aggregator, and `FINAL()` answer locking)
7. **Layer 7: Control** (Step limits, budget tracking, and recursion guards)
8. **Layer 8: Evaluation** (Metrics and benchmark scaffolds)

## 🛠️ Testing & Development

### Run all tests
```bash
pytest -q
```

### Layer-by-Layer Verification
Each layer has a dedicated test suite under `tests/`. For example, to verify the execution environment:
```bash
pytest -q tests/test_execution.py
```

---

## ✅ Completed Milestones
- [x] **Core REPL Loop**: Implemented `run_until_done` to support multi-round reasoning.
- [x] **Stateful Execution**: Context is mounted as a `context` variable; `print()` output is captured and fed back to the LLM.
- [x] **Recursive Tooling**: `llm_query()` is fully wired into the sandbox via `RecursionManager`.
- [x] **Groq Integration**: Support for `llama-3.3-70b` (Root) and `llama-3.1-8b` (Sub-call) models.
- [x] **Sandboxing**: Restricted builtins and step limiters are active by default.

## ⏳ Pending Tasks & Opportunities
- [ ] **Benchmark Loaders**: Implement full data loaders for S-NIAH, OOLONG, and BrowseComp in Layer 8.
- [ ] **Dynamic Cost Tracking**: Wire `BudgetManager` to actual token counts returned by the LLM metadata.
- [ ] **Async Orchestration**: Implement non-blocking `llm_query` calls for parallel context processing.
- [ ] **Layer 5 Integration**: Formally expose `peek_head`, `by_keyword`, and `chunker` as registered tools in the REPL.

---

*This project is built for experimentation. See `CONTRIBUTING.md` (or the layer sections in this README) for details on how to extend the controller or add new tools to the sandbox.*
