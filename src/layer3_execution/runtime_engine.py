"""Execute generated Python with validation, state persistence, and timeouts."""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import traceback
from typing import Any

from layer3_execution.sandbox import safe_builtins, validate_code
from layer3_execution.state_store import StateStore
from layer3_execution.tool_interface import ToolInterface
from layer7_control.step_limiter import StepLimiter
from layer7_control.budget_manager import BudgetManager
from layer7_control.token_tracker import TokenTracker
from shared.types import ExecutionResult


def _execute_worker(
    code: str,
    state_data: dict[str, Any],
    tool_namespace: dict[str, Any],
    strict_sandbox: bool,
    protected_names: set[str],
    budget: BudgetManager | None,
    token_tracker: TokenTracker | None,
    connection: Any,
) -> None:
    """Run one generated program in a killable child process."""
    buf = io.StringIO()
    budget_before = budget.used if budget else 0
    events_before = len(token_tracker.counts) if token_tracker else 0

    def usage() -> tuple[int, list[int]]:
        budget_delta = max(0, (budget.used if budget else 0) - budget_before)
        events = token_tracker.since(events_before) if token_tracker else []
        return budget_delta, events

    try:
        if strict_sandbox:
            validate_code(code)
        globals_: dict[str, Any] = {"__builtins__": safe_builtins() if strict_sandbox else __builtins__}
        globals_.update(state_data)
        globals_.update(tool_namespace)
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<generated>", "exec"), globals_, globals_)  # noqa: S102

        tool_names = set(tool_namespace)
        updates = {
            key: value
            for key, value in globals_.items()
            if not key.startswith("_")
            and key not in protected_names
            and key not in tool_names
            and not (callable(value) and key in tool_names)
        }
        budget_delta, events = usage()
        connection.send((True, globals_.get("result"), updates, None, buf.getvalue(), budget_delta, events))
    except Exception as exc:  # noqa: BLE001 - return generated-code failures
        budget_delta, events = usage()
        connection.send((False, None, {}, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", buf.getvalue(), budget_delta, events))
    finally:
        connection.close()


class RuntimeEngine:
    def __init__(
        self,
        state: StateStore,
        tools: ToolInterface,
        *,
        step_limiter: StepLimiter | None = None,
        strict_sandbox: bool = True,
        timeout_seconds: float | None = 30.0,
        protected_names: set[str] | None = None,
        budget: BudgetManager | None = None,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.state = state
        self.tools = tools
        self.step_limiter = step_limiter
        self.strict_sandbox = strict_sandbox
        self.timeout_seconds = timeout_seconds
        self.protected_names = protected_names or set()
        self.budget = budget
        self.token_tracker = token_tracker

    def _build_globals(self) -> dict[str, Any]:
        globals_: dict[str, Any] = {
            "__builtins__": safe_builtins() if self.strict_sandbox else __builtins__
        }
        globals_.update(self.state.as_dict())
        globals_.update(self.tools.as_namespace())
        return globals_

    def _result_from_payload(
        self, payload: tuple[Any, ...], *, apply_usage: bool = True
    ) -> ExecutionResult:
        ok, value, updates, error, stdout, budget_delta, token_events = payload
        if apply_usage and self.budget and budget_delta:
            self.budget.spend(budget_delta)
        if apply_usage and self.token_tracker and token_events:
            self.token_tracker.record_many(token_events)
        if ok:
            for key, item in updates.items():
                self.state.set(key, item)
        return ExecutionResult(
            ok=ok,
            value=value,
            error=error,
            locals_snapshot=self.state.as_dict(),
            stdout=stdout,
        )

    def execute(self, code: str) -> ExecutionResult:
        try:
            if self.step_limiter is not None:
                self.step_limiter.tick()
            if self.strict_sandbox:
                validate_code(code)
        except Exception as exc:  # noqa: BLE001 - expose controlled failures
            return ExecutionResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        if self.timeout_seconds is None:
            parent, child = multiprocessing.Pipe(duplex=False)
            _execute_worker(
                code,
                self.state.as_dict(),
                self.tools.as_namespace(),
                self.strict_sandbox,
                self.protected_names,
                self.budget,
                self.token_tracker,
                child,
            )
            child.close()
            return self._result_from_payload(parent.recv(), apply_usage=False)

        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_execute_worker,
            args=(
                code,
                self.state.as_dict(),
                self.tools.as_namespace(),
                self.strict_sandbox,
                self.protected_names,
                self.budget,
                self.token_tracker,
                child,
            ),
        )
        process.start()
        child.close()
        try:
            if not parent.poll(self.timeout_seconds):
                process.terminate()
                process.join()
                return ExecutionResult(
                    ok=False,
                    error=f"TimeoutError: execution exceeded {self.timeout_seconds:g} seconds",
                )
            return self._result_from_payload(parent.recv())
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            parent.close()
