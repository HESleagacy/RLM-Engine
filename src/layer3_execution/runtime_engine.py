"""Execute generated Python; does not decide — only runs.

Gap 3 fix: stdout from print() inside exec() is now captured via
contextlib.redirect_stdout and returned in ExecutionResult.stdout.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any

from layer3_execution.sandbox import safe_builtins
from layer3_execution.state_store import StateStore
from layer3_execution.tool_interface import ToolInterface
from layer7_control.step_limiter import StepLimiter
from shared.types import ExecutionResult


class RuntimeEngine:
    def __init__(
        self,
        state: StateStore,
        tools: ToolInterface,
        *,
        step_limiter: StepLimiter | None = None,
        strict_sandbox: bool = True,
    ) -> None:
        self.state = state
        self.tools = tools
        self.step_limiter = step_limiter
        self.strict_sandbox = strict_sandbox

    def _build_globals(self) -> dict[str, Any]:
        g: dict[str, Any] = {}
        g["__builtins__"] = safe_builtins() if self.strict_sandbox else __builtins__
        g.update(self.state.as_dict())
        g.update(self.tools.as_namespace())
        return g

    def execute(self, code: str) -> ExecutionResult:
        if self.step_limiter is not None:
            self.step_limiter.tick()

        g = self._build_globals()
        buf = io.StringIO()

        try:
            # Gap 3: redirect stdout so print() output is captured
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<generated>", "exec"), g, g)  # noqa: S102
        except Exception as e:  # noqa: BLE001 — surface sandbox errors
            return ExecutionResult(
                ok=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                locals_snapshot={},
                stdout=buf.getvalue(),
            )

        # Persist user-defined names back into state (exclude builtins noise)
        for k, v in g.items():
            if k.startswith("_") or k in ("__builtins__",):
                continue
            if callable(v) and k in self.tools.as_namespace():
                continue
            self.state.set(k, v)

        return ExecutionResult(
            ok=True,
            value=g.get("result"),
            locals_snapshot=self.state.as_dict(),
            stdout=buf.getvalue(),
        )
