"""Execute generated Python with validation, state persistence, and timeouts."""

from __future__ import annotations

import contextlib
import multiprocessing
from multiprocessing.connection import wait
import traceback
import time
from typing import Any

from layer3_execution.sandbox import safe_builtins, validate_code
from layer3_execution.state_store import StateStore
from layer3_execution.tool_interface import ToolInterface
from layer7_control.step_limiter import StepLimiter
from layer7_control.budget_manager import BudgetManager
from layer7_control.token_tracker import TokenTracker
from shared.types import ExecutionResult
from shared.resource_errors import resource_error
from shared.model_errors import ModelError
from layer3_execution.stdout_buffer import BoundedStdout


def _execute_payload(
    code: str,
    state_data: dict[str, Any],
    tool_namespace: dict[str, Any],
    strict_sandbox: bool,
    protected_names: set[str],
    budget: BudgetManager | None,
    token_tracker: TokenTracker | None,
    evidence_ledger=None,
    stdout_limit=65_536,
) -> tuple[Any, ...]:
    """Run generated code and collect its result, or a small failure payload."""
    buf = BoundedStdout(stdout_limit)
    budget_before = budget.used if budget else 0
    uncertain_before = budget.uncertain if budget else 0
    events_before = len(token_tracker.counts) if token_tracker else 0

    def usage():
        budget_delta = max(0, (budget.used if budget else 0) - budget_before)
        uncertain_delta = max(0, (budget.uncertain if budget else 0) - uncertain_before)
        events = token_tracker.since(events_before) if token_tracker else []
        return (budget_delta, uncertain_delta), events

    def payload(ok, value, updates, error, code=None):
        budget_delta, events = usage()
        return (
            ok, value, updates, error, buf.getvalue(), budget_delta, events, code,
            evidence_ledger.issued() if evidence_ledger else (), buf.truncated,
        )

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
        return payload(True, globals_.get("result"), updates, None)
    except Exception as exc:  # noqa: BLE001 - return generated-code failures
        failure = resource_error(exc)
        if failure is not None:
            # Avoid formatting tracebacks or copying stdout after allocation failure.
            try:
                budget_delta, events = usage()
            except MemoryError:
                budget_delta, events = (0, 0), []
            return (False, None, {}, str(failure), "", budget_delta, events, failure.code, (), buf.truncated)
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return payload(False, None, {}, error[:4096], exc.code if isinstance(exc, ModelError) else "execution_error")


def _execute_worker(*args: Any) -> None:
    """Send a payload from a killable worker; serialization can fail too."""
    *execution_args, connection = args
    result = None
    try:
        result = _execute_payload(*execution_args)
        connection.send(result)
    except Exception as exc:
        failure = resource_error(exc)
        connection.send((
            False, None, {}, str(failure) if failure else f"{type(exc).__name__}: {exc}",
            "", result[5] if result else (0, 0), result[6] if result else [],
            failure.code if failure else "execution_error", (), False,
        ))
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
        before_step=None,
        evidence_ledger=None,
        stdout_limit: int = 65_536,
        cancel_event=None,
    ) -> None:
        self.state = state
        self.tools = tools
        self.step_limiter = step_limiter
        self.strict_sandbox = strict_sandbox
        self.timeout_seconds = timeout_seconds
        self.protected_names = protected_names or set()
        self.budget = budget
        self.token_tracker = token_tracker
        self.before_step = before_step
        self.evidence_ledger = evidence_ledger
        self.stdout_limit = stdout_limit
        self.cancel_event = cancel_event

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
        ok, value, updates, error, stdout, budget_delta, token_events, error_code, evidence, truncated = payload
        if apply_usage and self.budget:
            self.budget.record_external(*budget_delta)
        if apply_usage and self.token_tracker and token_events:
            self.token_tracker.record_many(token_events)
        if self.evidence_ledger:
            self.evidence_ledger.ingest(evidence)
        if self.budget and self.budget.used > self.budget.limit:
            ok, error, error_code = False, "Token budget exhausted after provider usage", "budget_exhausted"
            value = None
        if ok:
            for key, item in updates.items():
                self.state.set(key, item)
        return ExecutionResult(
            ok=ok,
            value=value,
            error=error,
            locals_snapshot=self.state.as_dict(),
            stdout=stdout,
            error_code=error_code,
            evidence_records=evidence,
            stdout_truncated=truncated,
        )

    def execute(self, code: str) -> ExecutionResult:
        try:
            if self.before_step:
                self.before_step()
            if self.evidence_ledger:
                self.evidence_ledger.begin_step()
            return self._execute(code)
        except Exception as exc:
            failure = resource_error(exc)
            return ExecutionResult(
                ok=False,
                error=str(failure) if failure else f"{type(exc).__name__}: {exc}",
                error_code=failure.code if failure else exc.code if isinstance(exc, ModelError) else "execution_error",
            )

    def _execute(self, code: str) -> ExecutionResult:
        try:
            if self.step_limiter is not None:
                self.step_limiter.tick()
            if self.strict_sandbox:
                validate_code(code)
        except Exception as exc:  # noqa: BLE001 - expose controlled failures
            failure = resource_error(exc)
            return ExecutionResult(
                ok=False, error=str(failure) if failure else f"{type(exc).__name__}: {exc}",
                error_code=failure.code if failure else exc.code if isinstance(exc, ModelError) else "execution_error",
            )

        if self.cancel_event is not None and self.cancel_event.is_set():
            return ExecutionResult(ok=False, error="Execution cancelled", error_code="cancelled")
        if self.timeout_seconds is None:
            # No IPC in inline mode: sending a large payload to ourselves can block.
            payload = _execute_payload(
                code,
                self.state.as_dict(),
                self.tools.as_namespace(),
                self.strict_sandbox,
                self.protected_names,
                self.budget,
                self.token_tracker,
                self.evidence_ledger,
                self.stdout_limit,
            )
            return self._result_from_payload(payload, apply_usage=False)

        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe(duplex=False)
        process = None
        started = False
        try:
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
                    self.evidence_ledger,
                    self.stdout_limit,
                    child,
                ),
            )
            process.start()
            started = True
            child.close()
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    return ExecutionResult(ok=False, error="Execution cancelled", error_code="cancelled")
                remaining = max(0, deadline - time.monotonic())
                ready = wait(
                    [parent, process.sentinel],
                    timeout=min(0.1, remaining) if self.cancel_event is not None else remaining,
                )
                if ready:
                    break
                if time.monotonic() >= deadline:
                    return ExecutionResult(
                        ok=False,
                        error=f"TimeoutError: execution exceeded {self.timeout_seconds:g} seconds",
                        error_code="execution_timeout",
                    )
            if parent in ready or parent.poll():
                try:
                    return self._result_from_payload(parent.recv())
                except (EOFError, ConnectionError):
                    pass
            process.join(timeout=0.2)
            return ExecutionResult(
                ok=False,
                error=(
                    f"Execution worker terminated without returning a result "
                    f"(exit code {process.exitcode}). The OS may have killed it or it may "
                    "have crashed; memory exhaustion cannot be confirmed."
                ),
                error_code="worker_terminated",
            )
        finally:
            child.close()
            if process is not None:
                if started:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                process.close()
            parent.close()
