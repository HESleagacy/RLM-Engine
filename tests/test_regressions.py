import time
import hashlib
import json

import pytest

from layer1_input import MountedContext
from layer2_controller.code_generator import CodeGenerator
from layer2_controller.protocol import Action, EvidenceRef, parse_action
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


@pytest.mark.parametrize("file_backed", [False, True], ids=["ram", "disk"])
@pytest.mark.parametrize("timeout", [None, 5], ids=["inline", "worker"])
def test_temporary_rebinding_does_not_replace_mounted_context_or_query(tmp_path, file_backed, timeout):
    """Rebinding is intentionally allowed locally, but never persisted."""
    path = tmp_path / "context.txt"
    path.write_text("original", encoding="utf-8")
    context = (
        MountedContext.from_file(path, memory_threshold_bytes=0)
        if file_backed else MountedContext("original")
    )
    state = StateStore()
    state.set("context", context.text)
    state.set("query", "original query")
    runtime = RuntimeEngine(
        state, ToolInterface(), protected_names={"context", "query"},
        timeout_seconds=timeout,
    )

    result = runtime.execute(
        "context = 'rebound'\nquery = 'temporary query'\nresult = (context, query)"
    )

    assert result.ok
    assert result.value == ("rebound", "temporary query")
    assert state.get("context") is context.text
    assert state.get("query") == "original query"
    following_round = runtime.execute("result = (context[:], query)")
    assert following_round.ok
    assert following_round.value == ("original", "original query")
    assert path.read_text(encoding="utf-8") == "original"


def test_controller_parser_handles_parentheses_in_final_answer():
    result = parse_action(json.dumps({"action": "finalize", "text": "The answer is f(x) = 42."}))

    assert result.action == "finalize"
    assert result.text == "The answer is f(x) = 42."


def test_codegen_only_detects_real_result_assignment():
    def llm(_prompt: str, *, max_tokens: int | None = None, timeout_seconds=None) -> tuple[str, int]:
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
                return Action("execute", code="source = read_context(0, len(context))\nresult = query")
            return Action(
                "finalize_variable", name="result",
                evidence=(EvidenceRef(hashlib.sha256(b"context").hexdigest(), 0, 7),),
            )

    controller = RootController(
        RuntimeEngine(StateStore(), ToolInterface(), timeout_seconds=None),
        codegen=FakeCodeGenerator(),
        max_rounds=2,
    )
    context = MountedContext(text="context")

    first = controller.run_until_done(context, "first")
    second = controller.run_until_done(context, "second")

    assert first.require_answer() == "first"
    assert second.require_answer() == "second"
