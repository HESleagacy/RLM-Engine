"""CLI entry: load config, mount input, wire controller + execution."""

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
    groq_model: str = "llama-3.1-8b-instant",
) -> tuple[RootController, ExecutionMonitor]:
    cfg = load_yaml_config(config_path)
    ctrl = cfg.get("control", {})
    exec_cfg = cfg.get("execution", {})

    steps = StepLimiter(max_steps=int(ctrl.get("max_steps", 100)))
    budget = BudgetManager(limit=int(ctrl.get("token_budget", 100_000)))
    recursion = RecursionGuard(max_depth=int(ctrl.get("max_recursion_depth", 8)))
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
        from shared.groq_client import make_groq_llm

        codegen = CodeGenerator(make_groq_llm(model=groq_model))
    controller = RootController(runtime, codegen=codegen)
    return controller, monitor


def main() -> None:
    _load_dotenv_if_present()
    p = argparse.ArgumentParser(description="Layered Reasoning System")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    p.add_argument("--prompt", default="Hello", help="Raw prompt text (mounted as P)")
    p.add_argument(
        "--use-groq",
        action="store_true",
        help="Use Groq for code generation (requires GROQ_API_KEY and pip install -e \".[groq]\")",
    )
    p.add_argument(
        "--groq-model",
        default="llama-3.1-8b-instant",
        help="Groq model id when --use-groq",
    )
    args = p.parse_args()

    controller, monitor = build_system(
        args.config,
        use_groq=args.use_groq,
        groq_model=args.groq_model,
    )
    ctx = MountedContext(text=normalize(args.prompt))
    out = controller.run_round(ctx, instruction="Set result to a short echo of the prompt.")
    print("execution:", out)
    print("monitor:", monitor.snapshot())


if __name__ == "__main__":
    main()
