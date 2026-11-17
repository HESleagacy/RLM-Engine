"""Structured actions, lifecycle, provenance, bounded diagnostics, and stopping rules."""

import hashlib
import json
import threading
import importlib
import sys

import pytest

from layer1_input import MountedContext
from layer2_controller import Action, RootController, RunFailedError
from layer2_controller.code_generator import CodeGenerator
from layer2_controller.control_flow import ControlFlow, FlowState
from layer2_controller.evidence import EvidenceLedger
from layer2_controller.history import HistoryManager, summarize_state
from layer2_controller.protocol import parse_action, parse_legacy_action
from layer3_execution import RuntimeEngine, StateStore, ToolInterface
from layer7_control import BudgetManager, StepLimiter
from shared.model_call import RetryPolicy
from shared.model_errors import HistoryBudgetError, ProtocolError

pytestmark = pytest.mark.controller


def reference(text="It's blue.", start=0, end=None):
    return {
        "document_hash": hashlib.sha256(text.encode()).hexdigest(),
        "start_char": start, "end_char": len(text) if end is None else end,
    }


def execute(code="source = read_context(0, len(context))\nresult = source['text']", **kwargs):
    return {"action": "execute", "code": code, **kwargs}


def final(text="It's blue.", **kwargs):
    return {"action": "finalize", "text": text, "evidence": [reference(text)], **kwargs}


def controller_for(actions, *, text="It's blue.", max_steps=100, **kwargs):
    pending, messages = iter(actions), []

    def chat(payload, **request):
        messages.append(payload)
        action = next(pending)
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            action = action(payload, request)
        return (json.dumps(action) if isinstance(action, dict) else action), 1

    runtime = RuntimeEngine(
        StateStore(), ToolInterface(), timeout_seconds=None,
        step_limiter=StepLimiter(max_steps=max_steps), protected_names={"context", "query"},
    )
    generator = CodeGenerator(chat=chat, retry_policy=RetryPolicy(max_attempts=1))
    controller = RootController(runtime, codegen=generator, **kwargs)
    return controller, MountedContext(text), messages


@pytest.mark.parametrize("text", ["It's blue.", "f(x) = (42)", "世界 🙂", "FINAL_VAR(not_an_action)"])
def test_json_final_text_is_not_parsed_as_python_or_markers(text):
    controller, ctx, calls = controller_for([execute(), final(text)], text=text)
    outcome = controller.run_until_done(ctx, "Answer from the source")
    assert outcome.status == "completed"
    assert outcome.require_answer() == text
    assert outcome.verification == "provenance_validated"
    assert len(calls) == 2


@pytest.mark.parametrize("raw", [
    "FINAL(answer)",
    "```json\n{\"action\":\"finalize\",\"text\":\"answer\"}\n```",
    '[]', 'null',
    '{"action":"execute","code":"result=1","text":"premature"}',
    '{"action":"execute","code":"a","code":"b"}',
    '{"action":"finalize","text":"a","action":"execute","code":"b"}',
    '{"action":"execute","code":""}',
    '{"action":"execute","code":"x=1","phase":"invented"}',
    '{"action":"finalize_variable","name":"missing.name"}',
    '{"action":"finalize_variable","name":"_private"}',
    '{"action":"finalize_variable","name":"class"}',
    '{"action":"finalize","text":NaN}',
    '{"action":"finalize","text":"x"} {"action":"execute","code":"x=1"}',
])
def test_invalid_mixed_or_ambiguous_actions_are_rejected(raw):
    with pytest.raises(ProtocolError):
        parse_action(raw)


def test_protocol_rejects_oversized_and_wrongly_typed_evidence():
    with pytest.raises(ProtocolError):
        parse_action("x" * 65_537)
    for evidence in [None, {}, [{"document_hash": "a" * 64, "start_char": True, "end_char": 10}]]:
        with pytest.raises(ProtocolError):
            parse_action(json.dumps({"action": "finalize", "text": "a", "evidence": evidence}))


def test_legacy_parser_is_explicit_and_markers_inside_code_are_not_final():
    action = parse_legacy_action("```repl\nFINAL_VAR(answer)\n```")
    assert action.action == "execute"
    assert parse_legacy_action("FINAL(It's blue.)").text == "It's blue."
    with pytest.raises(ProtocolError):
        parse_legacy_action("```repl\nresult=1\n```\nFINAL_VAR(result)")


def test_legacy_controller_requires_explicit_protocol_and_unverified_policy():
    controller, ctx, _ = controller_for(["```repl\nresult = \"legacy\"\n```", "FINAL_VAR(result)"], require_evidence=False)
    controller.codegen.protocol = "legacy"
    outcome = controller.run_until_done(ctx, "legacy integration")
    assert outcome.require_answer() == "legacy"
    assert outcome.protocol == "legacy"
    assert outcome.verification == "unverified"


def test_invalid_response_is_repaired_without_executing_it():
    controller, ctx, calls = controller_for(["not JSON", execute(), final()])
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.ok
    assert controller.runtime.step_limiter.current == 1
    assert outcome.observations[0].error_code == "invalid_action"
    assert "invalid_action" in calls[1][-1]["content"]


def test_invalid_response_repair_budget_is_finite():
    controller, ctx, calls = controller_for(["bad", "bad", "bad"], max_repairs=1)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.status == "failed"
    assert outcome.error_code == "invalid_action"
    assert outcome.answer is None and len(calls) == 2
    with pytest.raises(RunFailedError):
        outcome.require_answer()


def test_missing_final_variable_is_a_failure_not_empty_success():
    controller, ctx, _ = controller_for([
        execute(), {"action": "finalize_variable", "name": "missing", "evidence": [reference()]},
    ], max_repairs=0)
    outcome = controller.run_until_done(ctx, "query")
    assert not outcome.ok
    assert outcome.error_code == "invalid_final"
    assert "does not exist" in outcome.message
    assert outcome.answer is None


@pytest.mark.parametrize("allow_empty", [False, True])
def test_empty_variable_is_distinct_from_missing_variable(allow_empty):
    controller, ctx, _ = controller_for([
        execute("read_context(0, len(context))\nresult = ''"),
        {"action": "finalize_variable", "name": "result", "evidence": [reference()]},
    ], allow_empty_final=allow_empty, max_repairs=0)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.ok is allow_empty
    assert outcome.answer == ("" if allow_empty else None)


def test_invalid_final_can_be_repaired_using_actual_state():
    controller, ctx, _ = controller_for([
        execute(),
        {"action": "finalize_variable", "name": "missing", "evidence": [reference()]},
        {"action": "finalize_variable", "name": "result", "evidence": [reference()]},
    ])
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.require_answer() == ctx.text
    assert outcome.observations[-1].error_code == "invalid_final"


def test_round_exhaustion_preserves_candidate_and_observations_but_not_an_answer():
    controller, ctx, _ = controller_for([execute("print('important evidence')\nresult='candidate'")], max_rounds=1)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.status == "incomplete" and outcome.stop_reason == "max_rounds"
    assert outcome.answer is None and outcome.partial_candidate == "candidate"
    assert outcome.observations[0].stdout_preview == "important evidence\n"


def test_finalize_before_inspection_is_rejected():
    controller, ctx, _ = controller_for([final()], max_repairs=0)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.error_code == "invalid_final"
    assert controller.runtime.step_limiter.current == 0


@pytest.mark.parametrize("ref", [
    {**reference(), "document_hash": "a" * 64},
    {**reference(), "end_char": 2},
    {**reference(), "end_char": 1000},
])
def test_wrong_document_unread_or_invalid_spans_cannot_support_finalization(ref):
    controller, ctx, _ = controller_for([execute(), {"action": "finalize", "text": "answer", "evidence": [ref]}], max_repairs=0)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.error_code == "invalid_final"
    assert outcome.answer is None


def test_forged_reference_is_rejected_even_when_source_span_exists():
    controller, ctx, _ = controller_for([execute("result = context[:]"), final()], max_repairs=0)
    outcome = controller.run_until_done(ctx, "query")
    assert "not read through" in outcome.message


def test_provenance_validation_does_not_claim_semantic_truth():
    controller, ctx, _ = controller_for([
        execute(), {"action": "finalize", "text": "An unsupported claim", "evidence": [reference()]},
    ])
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.verification == "provenance_validated"
    assert outcome.verification != "externally_verified"


@pytest.mark.parametrize("accepted", [True, False])
def test_external_verifier_is_optional_and_rejection_cannot_be_success(accepted):
    controller, ctx, _ = controller_for([execute(), final()], verifier=lambda *args: accepted, max_repairs=0)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.ok is accepted
    assert outcome.verification == ("externally_verified" if accepted else "unverified")


def test_external_verifier_failure_is_typed_and_not_exposed_as_an_answer():
    def broken(*args):
        raise RuntimeError("private verifier details")

    controller, ctx, _ = controller_for([execute(), final()], verifier=broken)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.error_code == "verification_failed"
    assert "private" not in outcome.message


def test_errors_survive_large_stdout_and_stdout_capture_is_bounded():
    controller, ctx, _ = controller_for([
        execute("print('x' * 200000)\nraise ValueError('critical failure')"),
    ], stdout_truncation=16, max_execution_failures=1)
    controller.runtime.stdout_limit = 32
    outcome = controller.run_until_done(ctx, "query")
    observation = outcome.observations[0]
    assert len(observation.stdout_preview) <= 16 and observation.stdout_truncated
    assert "critical failure" in observation.error_message
    assert outcome.status == "failed" and outcome.answer is None


def test_repeated_action_and_no_progress_are_bounded():
    controller, ctx, calls = controller_for([execute("print('same')")] * 10)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.status == "incomplete"
    assert outcome.stop_reason in ("repeated_action", "no_progress")
    assert len(calls) < 10


def test_repeated_execution_failures_do_not_run_until_all_rounds_are_spent():
    actions = [execute(f"raise ValueError('failure {index}')") for index in range(10)]
    controller, ctx, calls = controller_for(actions, max_execution_failures=2)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.stop_reason == "execution_failures"
    assert len(calls) == 2


def test_step_limit_still_allows_finalization_after_last_permitted_execution():
    controller, ctx, _ = controller_for([execute(), final()], max_steps=1)
    assert controller.run_until_done(ctx, "query").ok


def test_step_limit_rejects_further_execution_without_another_worker_step():
    controller, ctx, calls = controller_for([execute(), execute("result='extra'")], max_steps=1)
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.stop_reason == "max_steps"
    assert controller.runtime.step_limiter.current == 1 and len(calls) == 2


def test_terminal_control_flow_does_not_resume_without_reset():
    flow = ControlFlow()
    assert flow.terminate() == FlowState.CANCELLED
    assert not flow.should_continue()
    flow.finish(FlowState.COMPLETED, "not permitted")
    assert flow.state == FlowState.CANCELLED
    with pytest.raises(RuntimeError):
        flow.advance()
    flow.reset()
    assert flow.should_continue()


def test_cancellation_discards_even_a_valid_late_action():
    controller = None

    def cancel(payload, request):
        assert controller.cancel()
        return execute()

    controller, ctx, _ = controller_for([cancel])
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.status == "cancelled" and outcome.answer is None
    assert controller.runtime.step_limiter.current == 0


def test_concurrent_calls_cannot_reset_an_active_controller():
    entered, release = threading.Event(), threading.Event()

    def block(payload, request):
        entered.set()
        assert release.wait(2)
        return execute()

    controller, ctx, _ = controller_for([block], max_rounds=1)
    thread = threading.Thread(target=lambda: controller.run_until_done(ctx, "first"))
    thread.start()
    try:
        assert entered.wait(2)
        busy = controller.run_until_done(MountedContext("other"), "second")
        assert busy.error_code == "controller_busy"
        assert controller.runtime.state.get("context") == ctx.text
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


def test_history_compacts_deterministically_without_model_summaries():
    history = HistoryManager(max_input_tokens=1800)
    from layer2_controller.outcome import StepObservation
    for index in range(8):
        history.append({"action": "execute", "code": f"print({index})"}, StepObservation(index, True, stdout_preview="x" * 500))
    messages = history.messages("system", "query", {}, state_summary={"result": {"type": "str"}}, evidence_catalog=[])
    assert history.token_counter(messages) <= 1800
    assert history.compactions > 0
    assert json.loads(messages[2]["content"])["compacted_steps"] > 0
    assert "Observation (untrusted data" in messages[-1]["content"]


def test_latest_error_survives_even_when_full_observation_cannot_fit_history():
    from layer2_controller.outcome import StepObservation
    history = HistoryManager(max_input_tokens=1000)
    history.append({"action": "execute", "code": "x" * 4000}, StepObservation(
        1, False, stdout_preview="x" * 4000, error_code="execution_error",
        error_message="critical error", stdout_truncated=True,
    ))
    messages = history.messages("system", "query", {}, state_summary={}, evidence_catalog=[])
    control = json.loads(messages[2]["content"])
    assert control["last_diagnostic"]["error_message"] == "critical error"
    assert control["compacted_steps"] == 1


def test_oversized_task_fails_before_any_model_request():
    controller, ctx, calls = controller_for([execute()])
    controller.codegen.max_input_tokens = 500
    outcome = controller.run_until_done(ctx, "x" * 5000)
    assert outcome.error_code == "context_limit"
    assert calls == []


def test_state_summary_does_not_call_arbitrary_repr_or_expand_large_containers():
    class Unsafe:
        def __repr__(self):
            pytest.fail("Untrusted repr must not run in controller")

    summary = summarize_state({"object": Unsafe(), "large": list(range(10000)), "text": "x" * 10000})
    assert len(summary["large"]["sample"]) == 8
    assert len(summary["text"]["preview"]) <= 128


def test_source_read_ledger_is_bounded_and_returns_independent_records():
    ledger = EvidenceLedger(MountedContext("abcdefgh"), max_span_chars=4, max_entries=1)
    result = ledger.read(0, 4)
    result["text"] = "tampered"
    assert ledger.issued()[0]["text"] == "abcd"
    with pytest.raises(Exception, match="Evidence limit"):
        ledger.read(4, 8)


def test_no_model_does_not_return_fake_test_answer():
    controller = RootController(RuntimeEngine(StateStore(), ToolInterface(), timeout_seconds=None))
    outcome = controller.run_until_done(MountedContext("document"), "query")
    assert outcome.error_code == "model_not_configured"
    assert outcome.answer is None


def test_late_provider_response_returns_incomplete_not_success():
    clock = [0.0]

    def delayed(payload, request):
        assert request["timeout_seconds"] == 1
        clock[0] = 2
        return execute()

    controller, ctx, calls = controller_for([delayed], wall_clock_seconds=1, clock=lambda: clock[0])
    controller.codegen.clock = lambda: clock[0]
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.status == "incomplete" and outcome.error_code == "deadline_exceeded"
    assert len(calls) == 1 and controller.runtime.step_limiter.current == 0


def test_inline_execution_deadline_preserves_partial_candidate_without_finalizing():
    clock = [0.0]
    controller, ctx, _ = controller_for([
        execute("advance_time()\nresult = 'candidate'"),
    ], wall_clock_seconds=1, clock=lambda: clock[0])
    controller.codegen.clock = lambda: clock[0]
    controller.runtime.tools.register("advance_time", lambda: clock.__setitem__(0, 2))
    outcome = controller.run_until_done(ctx, "query")
    assert outcome.stop_reason == "deadline_exceeded"
    assert outcome.partial_candidate == "candidate" and outcome.answer is None


def test_changed_source_invalidates_even_previously_issued_evidence(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("It's blue.", encoding="utf-8")
    controller, _, _ = controller_for([execute(), lambda *args: (path.write_text("changed source"), final())[1]])
    outcome = controller.run_until_done(MountedContext.from_file(path, memory_threshold_bytes=0), "query")
    assert outcome.status == "failed"
    assert outcome.error_code == "source_unavailable"
    assert outcome.answer is None


@pytest.mark.parametrize("limit", [0, 1, 3, 4, 32])
def test_stdout_head_tail_capture_handles_fragmented_writes(limit):
    from layer3_execution.stdout_buffer import BoundedStdout
    buffer = BoundedStdout(limit)
    for part in ["ab", "c", "def", "g"]:
        assert buffer.write(part) == len(part)
    text = "abcdefg"
    head = (limit + 1) // 2
    expected = text if len(text) <= limit else text[:head] + (text[-(limit - head):] if limit > head else "")
    assert buffer.getvalue() == expected
    assert buffer.truncated == (len(text) > limit)


def test_cancellation_terminates_an_active_execution_worker(monkeypatch):
    import layer3_execution.runtime_engine as module
    entered = threading.Event()
    original_wait = module.wait

    def waiting(*args, **kwargs):
        entered.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(module, "wait", waiting)
    controller, ctx, _ = controller_for([execute("while True:\n    pass")])
    controller.runtime.timeout_seconds = 10
    results = []
    thread = threading.Thread(target=lambda: results.append(controller.run_until_done(ctx, "query")))
    thread.start()
    try:
        assert entered.wait(2)
        assert controller.cancel()
    finally:
        controller.cancel()
        thread.join(3)
    assert not thread.is_alive()
    assert results[0].status == "cancelled"
    assert results[0].answer is None


def test_source_identification_obeys_run_deadline():
    from shared.model_errors import DeadlineExceededError
    with pytest.raises(DeadlineExceededError):
        EvidenceLedger(MountedContext("source"), deadline=1, clock=lambda: 2)


@pytest.mark.parametrize(("status", "code", "http_status"), [
    ("failed", "out_of_memory", 503),
    ("failed", "rate_limited", 429),
    ("failed", "provider_timeout", 504),
    ("failed", "invalid_action", 502),
    ("incomplete", "max_rounds", 422),
])
def test_demo_reports_typed_run_outcomes_without_scoring_them(monkeypatch, status, code, http_status):
    from layer2_controller.outcome import RunOutcome
    demo = importlib.import_module("demo.app")

    def fail(*args):
        RunOutcome(status, code, error_code=code, message="Run did not complete").require_answer()

    monkeypatch.setattr(demo, "run_rlm", fail)
    with demo.app.test_client() as client:
        response = client.post("/api/run", json={"document": "Hero Arjun", "facts": {"Hero": "Arjun"}})
    assert response.status_code == http_status
    body = response.get_json()
    assert body["code"] == code and body["status"] == status
    assert "story" not in body and "score" not in body


def test_cli_incomplete_run_exits_nonzero_and_never_prints_a_final_answer(monkeypatch, capsys):
    from layer2_controller.outcome import RunOutcome
    main = importlib.import_module("main")
    controller = type("Incomplete", (), {
        "run_until_done": lambda *args: RunOutcome("incomplete", "max_rounds", partial_candidate="not final")
    })()
    monkeypatch.setattr(main, "build_system", lambda *args, **kwargs: (controller, None))
    monkeypatch.setattr(sys, "argv", ["rlm", "--prompt", "document", "--use-groq", "--query", "question"])
    with pytest.raises(SystemExit) as caught:
        main.main()
    output = capsys.readouterr()
    assert caught.value.code == 1
    assert "[max_rounds]" in output.err
    assert "Final Answer" not in output.out and "not final" not in output.out


@pytest.mark.parametrize("fields", [
    {"status": "running", "stop_reason": "not final"},
    {"status": "completed", "stop_reason": "finalized"},
    {"status": "failed", "stop_reason": "failure", "answer": "not a valid answer"},
])
def test_typed_outcomes_reject_inconsistent_success_failure_contracts(fields):
    from layer2_controller.outcome import RunOutcome
    with pytest.raises(ValueError):
        RunOutcome(**fields)


@pytest.mark.parametrize("query", [None, "", 42])
def test_invalid_task_is_rejected_before_model_requests(query):
    controller, ctx, calls = controller_for([execute()])
    outcome = controller.run_until_done(ctx, query)
    assert outcome.error_code == "invalid_task"
    assert outcome.answer is None and not calls


@pytest.mark.parametrize("flags", [
    ["--query", "question"], ["--query", "", "--use-groq"],
])
def test_cli_does_not_silently_echo_a_document_in_place_of_answering_query(monkeypatch, capsys, flags):
    main = importlib.import_module("main")
    monkeypatch.setattr(sys, "argv", ["rlm", "--prompt", "document", *flags])
    with pytest.raises(SystemExit) as caught:
        main.main()
    output = capsys.readouterr()
    assert caught.value.code == 2
    assert "--query" in output.err and output.out == ""
