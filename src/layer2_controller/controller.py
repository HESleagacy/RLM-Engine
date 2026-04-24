"""Root controller — orchestrates planning, REPL loop, and execution.

Gaps fixed here:
  Gap 1 — mount `context` variable in REPL globals before first execute()
  Gap 2 — llm_query is registered externally (in build_system); controller consumes it
  Gap 4 — run_until_done() drives the multi-round REPL loop
  Gap 5 — FINAL() / FINAL_VAR() detection terminates the loop
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from layer2_controller.code_generator import CodeGenerator, StepResult
from layer2_controller.control_flow import ControlFlow
from layer2_controller.planner import PlanAction, Planner
from layer3_execution.runtime_engine import RuntimeEngine
from layer6_output.finalizer import FINAL, FinalAnswer
from layer6_output.output_manager import OutputManager

if TYPE_CHECKING:
    from layer1_input.context_repr import MountedContext

_STDOUT_TRUNCATION = 3000  # chars of REPL output fed back to root LLM per round


class RootController:
    def __init__(
        self,
        runtime: RuntimeEngine,
        *,
        planner: Planner | None = None,
        codegen: CodeGenerator | None = None,
        flow: ControlFlow | None = None,
        output: OutputManager | None = None,
        max_rounds: int = 20,
        stdout_truncation: int = _STDOUT_TRUNCATION,
    ) -> None:
        self.runtime = runtime
        self.planner = planner or Planner()
        self.codegen = codegen or CodeGenerator()
        self.flow = flow or ControlFlow()
        self.output = output or OutputManager()
        self.max_rounds = max_rounds
        self.stdout_truncation = stdout_truncation

    # ── Legacy single-round method (all existing tests use this) ──────────────

    def run_round(self, ctx: "MountedContext", instruction: str) -> dict[str, Any]:
        """Single round: plan → codegen → exec → record. Used by tests."""
        plan = self.planner.next_step(self.flow.iteration, has_more=True)
        if plan.action == PlanAction.STOP:
            return {"ok": False, "reason": "stop"}
        code = self.codegen.generate(instruction, ctx.text[:2000])
        result = self.runtime.execute(code)
        self.flow.advance()
        if result.ok and result.value is not None:
            self.output.intermediate.append(result.value)
        return {"ok": result.ok, "execution": result}

    # ── Gap 4 & 5: Full RLM REPL loop ─────────────────────────────────────────

    def run_until_done(self, ctx: "MountedContext", query: str) -> FinalAnswer:
        """
        Primary RLM entry point for long-context reasoning.

        1. Gap 1 — mounts ctx.text as `context` in the REPL state
        2. Iterates up to max_rounds, each round calling generate_step()
        3. Gap 5 — stops on FINAL() / FINAL_VAR() in the LLM response
        4. Executes ```repl blocks and feeds stdout back to the LLM
        5. Returns a locked FinalAnswer
        """
        # Gap 1: mount context as a variable visible inside exec()
        self.runtime.state.set("context", ctx.text)

        history: list[dict[str, str]] = []

        for round_idx in range(self.max_rounds):
            # Gap 6: generate_step uses the paper's system prompt + chat history
            step: StepResult = self.codegen.generate_step(
                query=query,
                context_total_length=len(ctx.text),
                history=history,
            )

            # Gap 5: FINAL_VAR — resolve variable name from REPL state
            if step.is_final and step.final_var is not None:
                val = self.runtime.state.as_dict().get(step.final_var, "")
                return FINAL(str(val))

            # Gap 5: FINAL(text) — direct answer
            if step.is_final and step.final_text is not None:
                return FINAL(step.final_text)

            # No code and no FINAL — LLM gave up or hallucinated; stop cleanly
            if step.code is None:
                break

            # Execute the repl code block
            result = self.runtime.execute(step.code)
            self.flow.advance()

            # Collect intermediate result values
            if result.ok and result.value is not None:
                self.output.intermediate.append(result.value)

            # Gap 3: build REPL output for next-round history (stdout + errors)
            repl_output = result.stdout or ""
            if not result.ok and result.error:
                repl_output += f"\n[REPL ERROR] {result.error}"
            # Truncate to prevent context explosion
            repl_output = repl_output[: self.stdout_truncation]

            history.append({"code": step.code, "output": repl_output})

        # Exhausted max_rounds — return best accumulated intermediate result
        return self.output.finalize_joined()
