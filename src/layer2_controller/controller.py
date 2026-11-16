"""Typed, bounded reasoning lifecycle with explicit failures and provenance."""

from __future__ import annotations

from collections import deque
import json
import math
import threading
import time

from layer2_controller.code_generator import CodeGenerator
from layer2_controller.control_flow import ControlFlow, FlowState
from layer2_controller.evidence import EvidenceLedger
from layer2_controller.history import HistoryManager, preview, summarize_state
from layer2_controller.outcome import RunOutcome, StepObservation
from layer2_controller.planner import PlanAction, Planner
from layer2_controller.protocol import validate_action
from layer6_output.output_manager import OutputManager
from shared.model_errors import EvidenceError, InvalidTaskError, ModelError, ProtocolError, VerificationError
from shared.model_call import estimate_input_tokens
from shared.resource_errors import WorkerTerminatedError, report_resource_errors, resource_error


def answer_text(value, *, allow_empty=False):
    if type(value) is str:
        text = value
    elif type(value) in (int, float, bool, list, tuple, dict):
        pieces, size = [], 0
        for piece in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
            size += len(piece)
            if size > 16_384:
                raise EvidenceError("Final value exceeds 16384 characters")
            pieces.append(piece)
        text = "".join(pieces)
    else:
        raise EvidenceError("Final variable must contain text or a JSON-serializable value")
    if len(text) > 16_384 or (not allow_empty and not text.strip()):
        raise EvidenceError("Final answer is empty or too large")
    return text


class RootController:
    def __init__(
        self, runtime, *, planner=None, codegen=None, flow=None, output=None,
        max_rounds=20, stdout_truncation=3000, max_repairs=2,
        max_execution_failures=3, wall_clock_seconds=300,
        require_evidence=True, allow_empty_final=False, verifier=None,
        recursion_manager=None, clock=time.monotonic,
    ):
        if (
            any(type(value) is not int for value in (
                max_rounds, stdout_truncation, max_repairs, max_execution_failures
            )) or not 0 <= max_rounds <= 1000 or stdout_truncation < 0 or max_repairs < 0
            or max_execution_failures < 1 or not math.isfinite(wall_clock_seconds)
            or wall_clock_seconds <= 0
        ):
            raise ValueError("Invalid controller limits")
        self.runtime = runtime
        self.planner = planner or Planner()
        self.codegen = codegen or CodeGenerator()
        self.flow = flow or ControlFlow(max_iterations=max_rounds)
        self.output = output or OutputManager()
        self.max_rounds, self.stdout_truncation = max_rounds, stdout_truncation
        self.max_repairs, self.max_execution_failures = max_repairs, max_execution_failures
        self.wall_clock_seconds = wall_clock_seconds
        self.require_evidence, self.allow_empty_final = require_evidence, allow_empty_final
        self.verifier, self.recursion_manager = verifier, recursion_manager
        self.clock = clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._running = False

    def cancel(self):
        if not self._running:
            return False
        self._cancel.set()
        self.flow.terminate()
        return True

    @report_resource_errors
    def run_round(self, ctx, instruction):
        """Explicit legacy single-round API; no automatic conversion to a final answer."""
        plan = self.planner.next_step(self.flow.iteration, has_more=True)
        if plan.action == PlanAction.STOP or not self.flow.should_continue():
            return {"ok": False, "reason": "stop"}
        self.runtime.state.set("context", ctx.text)
        code = self.codegen.generate(instruction, ctx.text[:2000])
        result = self.runtime.execute(code)
        self.flow.advance()
        if result.ok and result.value is not None:
            self.output.intermediate.append(result.value)
        return {"ok": result.ok, "execution": result}

    def run_until_done(self, ctx, query) -> RunOutcome:
        if not self._lock.acquire(blocking=False):
            return RunOutcome("failed", "controller_busy", error_code="controller_busy", message="Controller is already running")
        self._cancel.clear()
        self._running = True
        try:
            return self._run(ctx, query)
        except Exception as exc:
            failure = resource_error(exc)
            code = failure.code if failure else exc.code if isinstance(exc, ModelError) else "controller_error"
            self.flow.finish(FlowState.FAILED, code)
            return RunOutcome("failed", code, error_code=code, message=str(failure) if failure else str(exc) if isinstance(exc, ModelError) else "Controller setup failed")
        finally:
            self._running = False
            self._lock.release()

    def _run(self, ctx, query):
        self.flow.reset()
        self.planner.reset()
        self.output.reset()
        self.runtime.state.clear()
        if not isinstance(query, str) or not query.strip():
            raise InvalidTaskError("Reasoning requires a nonempty text query")
        self.runtime.state.set("context", ctx.text)
        self.runtime.state.set("query", query)
        if isinstance(self.codegen, CodeGenerator):
            if self.codegen.budget is None:
                self.codegen.budget = self.runtime.budget
            elif self.runtime.budget is None:
                self.runtime.budget = self.codegen.budget
            elif self.codegen.budget is not self.runtime.budget:
                raise ValueError("Root and execution must share one system budget")
            if self.codegen.token_tracker is None:
                self.codegen.token_tracker = self.runtime.token_tracker
            elif self.runtime.token_tracker is None:
                self.runtime.token_tracker = self.codegen.token_tracker
            elif self.codegen.token_tracker is not self.runtime.token_tracker:
                raise ValueError("Root and execution must share one token tracker")
        self.history = HistoryManager(
            max_input_tokens=getattr(self.codegen, "max_input_tokens", 24_000),
            token_counter=getattr(self.codegen, "token_counter", None) or estimate_input_tokens,
        )
        observations = deque(maxlen=128)
        observation_count = repairs = failures = successful_steps = 0
        candidate = None
        ledger = None
        budget = self.runtime.budget or getattr(self.codegen, "budget", None)
        budget_before = budget.used if budget else 0
        uncertain_before = budget.uncertain if budget else 0
        deadline = self.clock() + self.wall_clock_seconds
        previous_ledger = self.runtime.evidence_ledger
        previous_reader = self.runtime.tools.as_namespace().get("read_context")
        previous_hook = self.runtime.before_step
        previous_cancel_event = self.runtime.cancel_event
        previous_deadline = getattr(self.recursion_manager, "deadline", None)
        if hasattr(self.codegen, "begin_run"):
            self.codegen.begin_run()

        def finish(state, reason, *, answer=None, error=None, message=None, evidence=(), verification="unverified"):
            self.flow.finish(state, reason)
            status = self.flow.state.value
            if status != "completed":
                answer = None
                reason = self.flow.stop_reason or reason
            if status == "cancelled":
                error, message = None, "Run cancelled"
            return RunOutcome(
                status, reason, answer=answer, partial_candidate=candidate,
                error_code=error, message=message,
                observations=tuple(observations), evidence=tuple(evidence),
                document_hash=ledger.document_hash if ledger else None,
                verification=verification if status == "completed" else "unverified",
                model_attempts=getattr(self.codegen, "attempts", 0),
                history_compactions=self.history.compactions,
                usage_uncertain=bool(
                    (budget and budget.uncertain > uncertain_before)
                    or getattr(self.codegen, "unknown_usage_events", 0)
                ),
                token_usage={
                    "run_charged": budget.used - budget_before,
                    "system_charged": budget.used,
                    "run_uncertain_charged": budget.uncertain - uncertain_before,
                } if budget else None,
                dropped_observations=max(0, observation_count - len(observations)),
                protocol=getattr(self.codegen, "protocol", "json"),
            )

        def observe(action, ok, *, stdout="", truncated=False, code=None, message=None, records=()):
            nonlocal observation_count
            observation = StepObservation(
                round=self.flow.iteration, ok=ok,
                stdout_preview=preview(stdout, self.stdout_truncation),
                stdout_truncated=truncated or len(stdout) > self.stdout_truncation,
                error_code=code, error_message=preview(message or "", 1024) or None,
                state_summary=summarize_state(self.runtime.state.as_dict()),
                evidence=tuple(
                    {
                        **{k: v for k, v in record.items() if k != "text"},
                        "text": preview(record["text"], 512),
                        "text_truncated": len(record["text"]) > 512,
                        "full_text_characters": len(record["text"]),
                    }
                    for record in records[:4]
                ),
            )
            observations.append(observation)
            observation_count += 1
            self.history.append(action, observation)
            return observation

        try:
            ledger = EvidenceLedger(ctx, deadline=deadline, clock=self.clock)
            self.runtime.evidence_ledger = ledger
            self.runtime.cancel_event = self._cancel
            self.runtime.tools.register("read_context", ledger.read)
            if self.recursion_manager is not None:
                self.recursion_manager.deadline = deadline
                self.runtime.before_step = self.recursion_manager.begin_step
            for _ in range(self.max_rounds):
                if self._cancel.is_set() or self.flow.state == FlowState.CANCELLED:
                    return finish(FlowState.CANCELLED, "cancelled", message="Run cancelled")
                if self.clock() >= deadline:
                    return finish(FlowState.INCOMPLETE, "deadline_exceeded", error="deadline_exceeded", message="Run deadline exceeded")
                if not self.flow.should_continue():
                    break
                steps = self.runtime.step_limiter
                if steps is not None and not steps.allow() and successful_steps == 0:
                    return finish(FlowState.INCOMPLETE, "max_steps", error="step_limit", message="Execution step limit reached")
                self.flow.advance()
                capabilities = {
                    "tools": sorted(self.runtime.tools.as_namespace()),
                    "max_subcalls_per_step": self.recursion_manager.max_subcalls_per_step if self.recursion_manager else 0,
                    "subcall_output_tokens": self.recursion_manager.output_tokens if self.recursion_manager else 0,
                    "remaining_tokens": budget.remaining() if budget else None,
                    "remaining_steps": steps.max_steps - steps.current if steps else None,
                    "remaining_rounds": self.max_rounds - self.flow.iteration,
                    "remaining_seconds": max(0, deadline - self.clock()),
                    "phase": self.planner.phase,
                    "require_evidence": self.require_evidence,
                    "allow_empty_final": self.allow_empty_final,
                    "max_evidence_span_chars": ledger.max_span_chars,
                    "max_evidence_entries": ledger.max_entries,
                    "remaining_evidence_slots": ledger.max_entries - len(ledger.records),
                    "max_final_references": 32,
                    "max_code_characters": 32_768,
                    "max_final_characters": 16_384,
                    "protocol": getattr(self.codegen, "protocol", "json"),
                }
                try:
                    action = validate_action(self.codegen.generate_step(
                        query=query, context_total_length=ledger.length, history=self.history,
                        capabilities=capabilities,
                        state_summary=summarize_state(self.runtime.state.as_dict()),
                        evidence_catalog=ledger.catalog(), deadline=deadline,
                    ))
                except ProtocolError as exc:
                    repairs += 1
                    observe({"invalid_action": True}, False, code=exc.code, message=str(exc))
                    if repairs > self.max_repairs:
                        return finish(FlowState.FAILED, "invalid_action", error=exc.code, message=str(exc))
                    continue
                if self._cancel.is_set():
                    return finish(FlowState.CANCELLED, "cancelled", message="Run cancelled")
                if self.clock() >= deadline:
                    return finish(FlowState.INCOMPLETE, "deadline_exceeded", error="deadline_exceeded", message="Run deadline exceeded")
                if action.action != "execute":
                    try:
                        if successful_steps == 0:
                            raise EvidenceError("Execute and inspect source data before finalizing")
                        if action.action == "finalize_variable":
                            namespace = self.runtime.state.as_dict()
                            if action.name not in namespace:
                                raise EvidenceError(f"Final variable '{action.name}' does not exist")
                            value = namespace[action.name]
                        else:
                            value = action.text
                        text = answer_text(value, allow_empty=self.allow_empty_final)
                        if self.require_evidence and not action.evidence:
                            raise EvidenceError("Document-grounded finalization requires source-read references")
                        records = ledger.validate(action.evidence)
                        verification = "provenance_validated" if records else "unverified"
                        if self.verifier is not None:
                            try:
                                accepted = self.verifier(text, records)
                            except Exception as exc:
                                if resource_error(exc):
                                    raise
                                raise VerificationError("External verifier failed") from exc
                            if accepted is not True:
                                raise EvidenceError("External verifier rejected the candidate")
                            verification = "externally_verified"
                        if self.clock() >= deadline:
                            return finish(FlowState.INCOMPLETE, "deadline_exceeded", error="deadline_exceeded", message="Run deadline exceeded during verification")
                        return finish(FlowState.COMPLETED, "finalized", answer=text, evidence=action.evidence, verification=verification)
                    except (EvidenceError, ValueError, TypeError) as exc:
                        repairs += 1
                        observe(action.to_dict(), False, code="invalid_final", message=str(exc))
                        if repairs > self.max_repairs:
                            return finish(FlowState.FAILED, "invalid_final", error="invalid_final", message=str(exc))
                        continue
                summary = summarize_state(self.runtime.state.as_dict())
                if steps is not None and not steps.allow():
                    return finish(FlowState.INCOMPLETE, "max_steps", error="step_limit", message="Execution step limit reached")
                if self.planner.repeated(action.code, summary, len(ledger.records)):
                    return finish(FlowState.INCOMPLETE, "repeated_action", error="repeated_action", message="Repeated action without new state or evidence")
                original_timeout = self.runtime.timeout_seconds
                try:
                    if original_timeout is not None:
                        self.runtime.timeout_seconds = min(original_timeout, max(0.001, deadline - self.clock()))
                    result = self.runtime.execute(action.code)
                finally:
                    self.runtime.timeout_seconds = original_timeout
                observation = observe(
                    action.to_dict(), result.ok, stdout=result.stdout,
                    truncated=result.stdout_truncated, code=result.error_code,
                    message=result.error, records=result.evidence_records,
                )
                if result.error_code in ("disk_full", "out_of_memory", "worker_terminated", "source_unavailable"):
                    return finish(FlowState.FAILED, result.error_code, error=result.error_code, message=result.error)
                if result.error_code in ("budget_exhausted", "step_limit", "deadline_exceeded"):
                    return finish(FlowState.INCOMPLETE, result.error_code, error=result.error_code, message=result.error)
                if result.ok:
                    successful_steps += 1
                    failures = 0
                    if result.value is not None:
                        try:
                            candidate = answer_text(result.value, allow_empty=True)
                        except (ValueError, TypeError, EvidenceError):
                            pass
                else:
                    failures += 1
                    if failures >= self.max_execution_failures:
                        return finish(FlowState.FAILED, "execution_failures", error=result.error_code, message=result.error)
                if self._cancel.is_set():
                    return finish(FlowState.CANCELLED, "cancelled", message="Run cancelled")
                if self.clock() >= deadline:
                    return finish(FlowState.INCOMPLETE, "deadline_exceeded", error="deadline_exceeded", message="Run deadline exceeded")
                stagnant = self.planner.observe(
                    ok=result.ok, state_summary=observation.state_summary,
                    evidence_count=len(ledger.records), has_candidate=candidate is not None,
                    phase=action.phase,
                )
                if stagnant:
                    return finish(FlowState.INCOMPLETE, "no_progress", error="no_progress", message="No relevant state or evidence progress")
            return finish(FlowState.INCOMPLETE, "max_rounds", error="max_rounds", message="Reasoning round limit reached")
        except ModelError as exc:
            state = FlowState.INCOMPLETE if exc.code in ("budget_exhausted", "deadline_exceeded", "context_limit") else FlowState.FAILED
            return finish(state, exc.code, error=exc.code, message=str(exc))
        except KeyboardInterrupt:
            return finish(FlowState.CANCELLED, "cancelled", message="Run interrupted")
        except Exception as exc:
            failure = resource_error(exc)
            code = failure.code if failure else "worker_terminated" if isinstance(exc, WorkerTerminatedError) else "controller_error"
            return finish(FlowState.FAILED, code, error=code, message=str(failure) if failure else str(exc) if isinstance(exc, WorkerTerminatedError) else "Controller operation failed")
        finally:
            self.runtime.evidence_ledger = previous_ledger
            self.runtime.before_step = previous_hook
            self.runtime.cancel_event = previous_cancel_event
            if previous_reader is not None:
                self.runtime.tools.register("read_context", previous_reader)
            else:
                self.runtime.tools.unregister("read_context")
            if self.recursion_manager is not None:
                self.recursion_manager.deadline = previous_deadline
