"""CLI entry: load config, mount input, wire controller + execution.

Gap 2 fix: RecursionManager is wired here and registered as `llm_query`
tool in ToolInterface so generated REPL code can call llm_query(...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from layer1_input.context_repr import MountedContext
from layer1_input.raw_loader import normalize
from layer2_controller.code_generator import CodeGenerator
from layer2_controller.controller import RootController
from layer3_execution.runtime_engine import RuntimeEngine
from layer3_execution.state_store import StateStore
from layer3_execution.tool_interface import ToolInterface
from layer4_recursion.recursion_manager import RecursionManager
from layer7_control.budget_manager import BudgetManager
from layer7_control.execution_monitor import ExecutionMonitor
from layer7_control.recursion_guard import RecursionGuard
from layer7_control.step_limiter import StepLimiter
from shared.utils import load_yaml_config


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


def build_system(
    config_path: str | Path,
    *,
    use_groq: bool = False,
    root_model: str = "llama-3.3-70b-versatile",
    sub_model: str = "llama-3.1-8b-instant",
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
    # Override models from config if not passed explicitly via CLI
    root_model = rlm_cfg.get("root_model", root_model)
    sub_model = rlm_cfg.get("sub_model", sub_model)

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
    )

    codegen: CodeGenerator | None = None

    if use_groq:
        from shared.groq_client import make_groq_chat, make_groq_llm

        # Sub-LLM: cheaper/faster model for recursive sub-calls
        sub_llm = make_groq_llm(model=sub_model)

        # Gap 2: wire llm_query tool → RecursionManager → sub-LLM
        # Generated REPL code calls llm_query("...") which hits sub_llm
        rec_manager = RecursionManager(guard=recursion, llm=sub_llm)
        tools.register("llm_query", rec_manager.run_subtask)

        # Root LLM: stronger model for the root controller
        root_llm = make_groq_llm(model=root_model)
        root_chat = make_groq_chat(model=root_model)
        codegen = CodeGenerator(llm=root_llm, chat=root_chat)

    controller = RootController(
        runtime,
        codegen=codegen,
        max_rounds=max_rounds,
        stdout_truncation=stdout_trunc,
    )
    return controller, monitor


def main() -> None:
    _load_dotenv_if_present()
    p = argparse.ArgumentParser(description="RLM — Recursive Language Model Engine")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    p.add_argument(
        "--prompt",
        default="Hello",
        help="Raw context text to mount as P (the long document/corpus)",
    )
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
        default="llama-3.3-70b-versatile",
        help="Groq model for the root LLM",
    )
    p.add_argument(
        "--sub-model",
        default="llama-3.1-8b-instant",
        help="Groq model for recursive sub-LLM calls (llm_query)",
    )
    args = p.parse_args()

    controller, monitor = build_system(
        args.config,
        use_groq=args.use_groq,
        root_model=args.root_model,
        sub_model=args.sub_model,
    )
    ctx = MountedContext(text=normalize(args.prompt))

    if args.query and args.use_groq:
        # Full RLM REPL loop
        print(f"[RLM] mounting context ({len(ctx.text):,} chars), query: {args.query!r}")
        answer = controller.run_until_done(ctx, args.query)
        print("\n── Final Answer ──")
        print(answer.text)
    else:
        # Legacy single-round mode (no --query or no --use-groq)
        out = controller.run_round(ctx, instruction="Set result to a short echo of the prompt.")
        print("execution:", out)
        print("monitor:", monitor.snapshot())


if __name__ == "__main__":
    main()
