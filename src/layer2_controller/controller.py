"""Root controller — orchestrates planning and execution hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from layer2_controller.code_generator import CodeGenerator
from layer2_controller.control_flow import ControlFlow
from layer2_controller.planner import PlanAction, Planner
from layer3_execution.runtime_engine import RuntimeEngine
from layer6_output.output_manager import OutputManager

if TYPE_CHECKING:
    from layer1_input.context_repr import MountedContext


class RootController:
    def __init__(
        self,
        runtime: RuntimeEngine,
        *,
        planner: Planner | None = None,
        codegen: CodeGenerator | None = None,
        flow: ControlFlow | None = None,
        output: OutputManager | None = None,
    ) -> None:
        self.runtime = runtime
        self.planner = planner or Planner()
        self.codegen = codegen or CodeGenerator()
        self.flow = flow or ControlFlow()
        self.output = output or OutputManager()

    def run_round(self, ctx: "MountedContext", instruction: str) -> dict[str, Any]:
        plan = self.planner.next_step(self.flow.iteration, has_more=True)
        if plan.action == PlanAction.STOP:
            return {"ok": False, "reason": "stop"}
        code = self.codegen.generate(instruction, ctx.text[:2000])
        result = self.runtime.execute(code)
        self.flow.advance()
        if result.ok and result.value is not None:
            self.output.intermediate.append(result.value)
        return {"ok": result.ok, "execution": result}
