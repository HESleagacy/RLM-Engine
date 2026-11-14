"""CLI entry: load config, mount input, wire controller + execution.

Gap 2 fix: RecursionManager is wired here and registered as `llm_query`
tool in ToolInterface so generated REPL code can call llm_query(...).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.resources
import json
import sys
from pathlib import Path

from layer1_input.context_repr import MountedContext
from layer1_input.raw_loader import normalize
from layer1_input.metadata_provider import describe
from layer2_controller.code_generator import CodeGenerator
from layer2_controller.controller import RootController
from layer3_execution.runtime_engine import RuntimeEngine
from layer3_execution.state_store import StateStore
from layer3_execution.tool_interface import ToolInterface
from layer4_recursion.recursion_manager import RecursionManager
from layer5_context_access import (
    by_keyword,
    by_regex,
    fixed_windows,
    iter_windows,
    ContextIndex,
    peek_head,
    peek_tail,
)
from layer7_control.budget_manager import BudgetManager
from layer7_control.execution_monitor import ExecutionMonitor
from layer7_control.recursion_guard import RecursionGuard
from layer7_control.step_limiter import StepLimiter
from shared.utils import load_yaml_config
from shared.resource_errors import (
    ResourceExhaustedError, WorkerTerminatedError, report_resource_errors, resource_error,
)


def _load_dotenv_if_present() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def default_config_path() -> str:
    source_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if source_path.is_file():
        return str(source_path)
    return str(importlib.resources.files("configs") / "default.yaml")


def build_system(
    config_path: str | Path,
    *,
    use_groq: bool = False,
    root_model: str | None = None,
    sub_model: str | None = None,
    context_index: ContextIndex | None = None,
) -> tuple[RootController, ExecutionMonitor]:
    cfg = load_yaml_config(config_path)
    ctrl_cfg = cfg.get("control", {})
    exec_cfg = cfg.get("execution", {})
    rlm_cfg = cfg.get("rlm", {})

    # Layer 7 — control limits
    max_steps = int(ctrl_cfg.get("max_steps", 100))
    max_depth = int(ctrl_cfg.get("max_recursion_depth", 5))
    token_budget = int(ctrl_cfg.get("token_budget", 100_000))

    # RLM loop config
    max_rounds = int(rlm_cfg.get("max_rounds", 20))
    stdout_trunc = int(rlm_cfg.get("stdout_truncation", 3000))
    root_model = root_model or rlm_cfg.get("root_model", "llama-3.3-70b-versatile")
    sub_model = sub_model or rlm_cfg.get("sub_model", "llama-3.1-8b-instant")
    max_subcalls = int(cfg.get("recursion", {}).get("max_subcalls_per_step", 4))

    steps = StepLimiter(max_steps=max_steps)
    budget = BudgetManager(limit=token_budget)
    recursion = RecursionGuard(max_depth=max_depth)
    monitor = ExecutionMonitor(steps=steps, budget=budget, recursion=recursion)

    state = StateStore()
    tools = ToolInterface()
    runtime = RuntimeEngine(
        state,
        tools,
        step_limiter=steps,
        strict_sandbox=bool(exec_cfg.get("sandbox_strict", True)),
        timeout_seconds=float(ctrl_cfg.get("wall_clock_seconds", 300)),
        protected_names={"context", "query"},
        budget=budget,
        token_tracker=monitor.tokens,
    )

    def wrap_tool(fn):
        def wrapper(*args, **kwargs):
            ctx_text = state.get("context", "")
            return fn(MountedContext(text=ctx_text), *args, **kwargs)
        return wrapper

    tools.register("peek_head", wrap_tool(peek_head))
    tools.register("peek_tail", wrap_tool(peek_tail))
    tools.register("by_keyword", wrap_tool(by_keyword))
    tools.register("by_regex", wrap_tool(by_regex))
    tools.register("chunker", wrap_tool(fixed_windows))
    tools.register("iter_chunks", wrap_tool(iter_windows))

    def check_index():
        if context_index is None:
            raise ValueError("No search index is mounted; enable --index")
        if state.get("context") is not context_index.context.text:
            raise ValueError("Search index does not match the mounted context; rebuild the index")
        return context_index

    tools.register("search_context", lambda query, limit=5: check_index().search(query, limit=limit))

    def context_info():
        if context_index is not None:
            return json.loads(json.dumps(check_index().metadata["structure"]))
        return asdict(describe(MountedContext(state.get("context", ""))))

    tools.register("context_info", context_info)

    codegen: CodeGenerator | None = None

    if use_groq:
        from shared.groq_client import make_groq_chat, make_groq_llm

        # Sub-LLM: cheaper/faster model for recursive sub-calls
        sub_llm = make_groq_llm(model=sub_model)

        # Gap 2: wire llm_query tool → RecursionManager → sub-LLM
        # Generated REPL code calls llm_query("...") which hits sub_llm
        rec_manager = RecursionManager(
            guard=recursion,
            llm=sub_llm,
            budget=budget,
            token_tracker=monitor.tokens,
            max_subcalls_per_step=max_subcalls,
        )
        tools.register("llm_query", rec_manager.run_subtask)

        # Root LLM: stronger model for the root controller
        root_llm = make_groq_llm(model=root_model)
        root_chat = make_groq_chat(model=root_model)
        codegen = CodeGenerator(
            llm=root_llm,
            chat=root_chat,
            budget=budget,
            token_tracker=monitor.tokens,
        )

    controller = RootController(
        runtime,
        codegen=codegen,
        max_rounds=max_rounds,
        stdout_truncation=stdout_trunc,
    )
    return controller, monitor


@report_resource_errors
def _main() -> None:
    _load_dotenv_if_present()
    p = argparse.ArgumentParser(description="RLM — Recursive Language Model Engine")
    p.add_argument(
        "--config",
        default=default_config_path(),
        help="Path to YAML config",
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--prompt",
        default="Hello",
        help="Raw context text to mount as P (the long document/corpus)",
    )
    source.add_argument(
        "--context-file", type=Path,
        help="UTF-8 document on any local drive (large files use disk-backed reads)",
    )
    p.add_argument(
        "--memory-threshold-bytes", type=int, default=1_048_576,
        help="Largest file to keep in RAM; use 0 to force non-empty files onto disk",
    )
    p.add_argument("--context-block-chars", type=int, default=65_536)
    p.add_argument("--context-cache-blocks", type=int, default=8)
    p.add_argument("--context-cache-bytes", type=int, default=4_194_304)
    p.add_argument("--index", action="store_true", help="Build/reuse a local full-text chunk index")
    p.add_argument("--index-dir", type=Path, default=Path.home() / ".cache" / "rlm-engine" / "indexes")
    p.add_argument("--index-chunk-chars", type=int, default=2048)
    p.add_argument("--index-overlap", type=int, default=128)
    p.add_argument("--index-cache-bytes", type=int, default=4_194_304)
    inspect = p.add_mutually_exclusive_group()
    inspect.add_argument("--search", help="Search locally and print JSON (implies --index; no LLM call)")
    inspect.add_argument("--show-metadata", action="store_true", help="Print document metadata as JSON without inference")
    p.add_argument(
        "--query",
        default=None,
        help="Question to answer over the mounted context (activates full RLM loop)",
    )
    p.add_argument(
        "--use-groq",
        action="store_true",
        help="Use Groq LLMs (requires GROQ_API_KEY and pip install -e \".[groq]\")",
    )
    p.add_argument(
        "--root-model",
        default=None,
        help="Groq model for the root LLM",
    )
    p.add_argument(
        "--sub-model",
        default=None,
        help="Groq model for recursive sub-LLM calls (llm_query)",
    )
    args = p.parse_args()

    try:
        if args.context_file is not None:
            ctx = MountedContext.from_file(
                args.context_file,
                memory_threshold_bytes=args.memory_threshold_bytes,
                block_chars=args.context_block_chars,
                cache_blocks=args.context_cache_blocks,
                max_cache_bytes=args.context_cache_bytes,
            )
        else:
            ctx = MountedContext(text=normalize(args.prompt))
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        failure = resource_error(exc)
        if failure is not None:
            raise failure from exc
        p.error(str(exc))

    context_index = None
    try:
        if args.index or args.search is not None:
            context_index = ContextIndex.build(
                ctx, args.index_dir, chunk_chars=args.index_chunk_chars,
                overlap=args.index_overlap, dram_cache_bytes=args.index_cache_bytes,
            )
        if args.show_metadata:
            print(json.dumps({
                "storage_backend": ctx.storage_backend,
                "structure": context_index.metadata["structure"] if context_index else asdict(describe(ctx)),
                "index": context_index.metadata if context_index else None,
            }, indent=2, ensure_ascii=False))
            return
        if args.search is not None:
            print(json.dumps({
                "matches": context_index.search(args.search),
                "cache": context_index.cache_info,
            }, indent=2, ensure_ascii=False))
            return
    except (OSError, ValueError, RuntimeError) as exc:
        failure = resource_error(exc)
        if failure is not None:
            raise failure from exc
        p.error(str(exc))

    controller, monitor = build_system(
        args.config,
        use_groq=args.use_groq,
        root_model=args.root_model,
        sub_model=args.sub_model,
        context_index=context_index,
    )

    if args.query and args.use_groq:
        # Full RLM REPL loop
        print(
            f"[RLM] mounting context ({len(ctx.text):,} chars, "
            f"{ctx.storage_backend}), query: {args.query!r}"
        )
        answer = controller.run_until_done(ctx, args.query)
        print("\n── Final Answer ──")
        print(answer.text)
    else:
        # Legacy single-round mode (no --query or no --use-groq)
        out = controller.run_round(ctx, instruction="Set result to a short echo of the prompt.")
        execution = out.get("execution")
        if execution is not None and execution.error_code in ("disk_full", "out_of_memory"):
            raise ResourceExhaustedError(execution.error_code, execution.error or "")
        if execution is not None and execution.error_code == "worker_terminated":
            raise WorkerTerminatedError(execution.error or "Execution worker terminated")
        print("execution:", out)
        print("monitor:", monitor.snapshot())


def main() -> None:
    try:
        _main()
    except (ResourceExhaustedError, WorkerTerminatedError) as exc:
        print(f"rlm: error: [{exc.code}] {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
