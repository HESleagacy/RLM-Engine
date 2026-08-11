import time

import pytest

from layer1_input import MountedContext
from layer2_controller.code_generator import CodeGenerator, StepResult, _parse_step
from layer2_controller.controller import RootController
from layer3_execution import RuntimeEngine, StateStore, ToolInterface
from layer7_control import BudgetManager, StepLimiter


def test_strict_runtime_rejects_imports_and_dunder_access():
    runtime = RuntimeEngine(StateStore(), ToolInterface())

    assert not runtime.execute("import os").ok
    assert not runtime.execute("result = context.__class__").ok


def test_runtime_returns_step_limit_failure_instead_of_raising():
    runtime = RuntimeEngine(
        StateStore(), ToolInterface(), step_limiter=StepLimiter(max_steps=0)
    )

    result = runtime.execute("result = 1")

    assert not result.ok
    assert "step limit exceeded" in (result.error or "")


def test_timeout_terminates_generated_code():
    runtime = RuntimeEngine(
        StateStore(), ToolInterface(), timeout_seconds=0.05
    )

    started = time.monotonic()
    result = runtime.execute("while True:\n    pass")

    assert not result.ok
    assert "TimeoutError" in (result.error or "")
    assert time.monotonic() - started < 2


def test_mounted_context_cannot_be_rebound():
    state = StateStore()
    state.set("context", "original")
    runtime = RuntimeEngine(
        state, ToolInterface(), protected_names={"context", "query"}
    )

    result = runtime.execute("context = 'rebound'\nresult = context")

    assert result.ok
    assert result.value == "rebound"
    assert state.get("context") == "original"


def test_controller_parser_handles_parentheses_in_final_answer():
    result = _parse_step("FINAL(The answer is f(x) = 42.)")

    assert result.is_final
    assert result.final_text == "The answer is f(x) = 42."


def test_codegen_only_detects_real_result_assignment():
    def llm(_prompt: str, *, max_tokens: int | None = None) -> tuple[str, int]:
        return ("# result is mentioned here\nvalue = 3", 1)

    code = CodeGenerator(llm).generate("test", "context")
    namespace: dict[str, object] = {}
    exec(compile(code, "<test>", "exec"), namespace, namespace)

    assert namespace["value"] == 3
    assert namespace["result"] is None


def test_budget_rejects_negative_spending():
    with pytest.raises(ValueError):
        BudgetManager(limit=10).spend(-1)


def test_sniah_generation_is_reproducible():
    from layer8_evaluation.benchmarks.s_niah import generate_sniah_tasks

    first = generate_sniah_tasks(3, seed=7)
    second = generate_sniah_tasks(3, seed=7)

    assert first == second


def test_controller_resets_state_between_invocations():
    class FakeCodeGenerator:
        def __init__(self):
            self.answer = 0

        def generate_step(self, **_kwargs):
            self.answer += 1
            if self.answer % 2:
                return StepResult(raw="code", code="result = query")
            return StepResult(raw="FINAL_VAR(result)", is_final=True, final_var="result")

    controller = RootController(
        RuntimeEngine(StateStore(), ToolInterface(), timeout_seconds=None),
        codegen=FakeCodeGenerator(),
        max_rounds=2,
    )
    context = MountedContext(text="context")

    first = controller.run_until_done(context, "first")
    second = controller.run_until_done(context, "second")

    assert first.text == "first"
    assert second.text == "second"
