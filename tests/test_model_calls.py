"""Budget reservations, known/unknown usage, retries, deadlines, and transport wiring."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from layer1_input import MountedContext
from layer2_controller import RootController
from layer2_controller.code_generator import CodeGenerator
from layer3_execution import RuntimeEngine, StateStore, ToolInterface
from layer4_recursion import RecursionManager
from layer7_control import BudgetManager, RecursionGuard
from layer7_control.token_tracker import TokenTracker
from shared.model_call import RetryPolicy, call_model, classify_provider_error
from shared.model_errors import (
    AuthenticationError, BudgetExceededError, DeadlineExceededError,
    InvalidUsageError, ModelError, ProviderRequestError, ProviderTimeoutError,
    ProviderUnavailableError, RateLimitError, SubcallLimitError,
)

pytestmark = pytest.mark.controller


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_preflight_budget_includes_input_and_reserved_output():
    budget = BudgetManager(9)
    calls = []
    with pytest.raises(BudgetExceededError):
        call_model(lambda *args, **kwargs: calls.append(True), "task", budget=budget, token_counter=lambda _: 10)
    assert not calls and budget.used == budget.reserved == 0


def test_known_usage_settles_reservation_and_records_overrun_before_stopping():
    budget, tracker = BudgetManager(40), TokenTracker()
    with pytest.raises(BudgetExceededError):
        call_model(lambda *args, **kwargs: ("answer", 50), "task", budget=budget, tracker=tracker, max_tokens=20, token_counter=lambda _: 10)
    assert budget.used == 50 and budget.reserved == 0
    assert tracker.total() == 50


def test_unknown_usage_is_conservatively_charged_not_reported_as_zero():
    budget, tracker = BudgetManager(100), TokenTracker()
    assert call_model(lambda *args, **kwargs: ("answer", None), "task", budget=budget, tracker=tracker, max_tokens=20, token_counter=lambda _: 10) == ("answer", None)
    assert budget.used == budget.uncertain == 30 and budget.reserved == 0
    assert tracker.counts == []


@pytest.mark.parametrize("tokens", [-1, True, "12", 1.5])
def test_invalid_provider_usage_is_not_allowed_to_bypass_accounting(tokens):
    budget = BudgetManager(100)
    with pytest.raises(InvalidUsageError):
        call_model(lambda *args, **kwargs: ("text", tokens), "task", budget=budget, max_tokens=20, token_counter=lambda _: 10)
    assert budget.used == budget.uncertain == 30 and budget.reserved == 0


def test_atomic_reservation_prevents_concurrent_oversubscription():
    budget = BudgetManager(50)
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return "answer", 5

    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(call_model, blocked, "task", budget=budget, max_tokens=20, token_counter=lambda _: 10)
        try:
            assert entered.wait(2)
            assert budget.reserved == 30
            with pytest.raises(BudgetExceededError):
                call_model(blocked, "task", budget=budget, max_tokens=20, token_counter=lambda _: 30)
        finally:
            release.set()
        assert pending.result(timeout=2) == ("answer", 5)
    assert budget.used == 5 and budget.reserved == 0


@pytest.mark.parametrize(("status", "expected"), [
    (401, AuthenticationError), (403, AuthenticationError),
    (400, ProviderRequestError), (429, RateLimitError),
    (500, ProviderUnavailableError), (503, ProviderUnavailableError),
])
def test_provider_errors_are_safe_typed_and_classified(status, expected):
    error = RuntimeError("secret provider request body")
    error.status_code = status
    result = classify_provider_error(error)
    assert isinstance(result, expected)
    assert "secret" not in str(result)


def test_retry_after_http_date_is_understood():
    error = RuntimeError()
    error.status_code = 429
    error.response = SimpleNamespace(headers={"retry-after": format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20))})
    delay = classify_provider_error(error).retry_after
    assert 0 < delay <= 20


def test_retry_owner_respects_attempt_budget_and_unknown_failure_cost():
    clock, budget, tracker = Clock(), BudgetManager(200), TokenTracker()
    calls = []

    def provider(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RateLimitError("limited", retry_after=0.25)
        return "answer", 5

    result = call_model(
        provider, "task", budget=budget, tracker=tracker, max_tokens=20,
        token_counter=lambda _: 10, deadline=10, clock=clock, sleep=clock.sleep,
        policy=RetryPolicy(max_attempts=3, backoff_seconds=0.1),
    )
    assert result == ("answer", 5) and len(calls) == 3
    assert clock.now == 0.5
    assert budget.used == 65 and budget.uncertain == 60 and budget.reserved == 0
    assert tracker.total() == 5


def test_retry_after_exceeding_policy_is_not_ignored_by_retrying_early():
    calls = []

    def provider(*args, **kwargs):
        calls.append(True)
        raise RateLimitError("limited", retry_after=60)

    with pytest.raises(RateLimitError):
        call_model(provider, "task", policy=RetryPolicy(max_delay_seconds=1))
    assert calls == [True]


def test_permanent_failure_is_never_retried():
    calls = []

    def provider(*args, **kwargs):
        calls.append(True)
        raise AuthenticationError("credentials invalid")

    with pytest.raises(AuthenticationError):
        call_model(provider, "task")
    assert calls == [True]


def test_deadline_bounds_attempt_timeout_and_rejects_late_responses():
    clock, budget = Clock(), BudgetManager(100)
    seen = []

    def provider(*args, **kwargs):
        seen.append(kwargs["timeout_seconds"])
        clock.now += 6
        return "late answer", 5

    with pytest.raises(DeadlineExceededError):
        call_model(provider, "task", deadline=5, clock=clock, budget=budget, max_tokens=20, token_counter=lambda _: 10)
    assert seen == [5]
    assert budget.used == 5 and budget.reserved == 0


def test_retry_does_not_sleep_past_total_deadline():
    clock = Clock()
    with pytest.raises(DeadlineExceededError):
        call_model(
            lambda *args, **kwargs: (_ for _ in ()).throw(ProviderTimeoutError("timeout")),
            "task", deadline=0.1, clock=clock, sleep=clock.sleep,
        )
    assert clock.now == 0


@pytest.mark.parametrize("settings", [
    {"max_attempts": 0}, {"max_attempts": 6}, {"backoff_seconds": -1},
    {"backoff_seconds": float("nan")}, {"request_timeout_seconds": float("inf")},
])
def test_invalid_retry_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        RetryPolicy(**settings)


def test_subcall_limit_resets_explicitly_for_inline_execution_steps():
    manager = RecursionManager(RecursionGuard(2), lambda *args, **kwargs: ("sub answer", 1), max_subcalls_per_step=1)
    tools = ToolInterface()
    tools.register("llm_query", manager.run_subtask)
    runtime = RuntimeEngine(StateStore(), tools, timeout_seconds=None, before_step=manager.begin_step)
    assert runtime.execute("result = llm_query('first')").ok
    assert runtime.execute("result = llm_query('second')").ok
    failed = runtime.execute("llm_query('one')\nllm_query('too many')")
    assert not failed.ok and failed.error_code == "subcall_limit"


def test_worker_subcall_usage_is_reconciled_including_unknown_charges():
    budget, tracker = BudgetManager(10000), TokenTracker()
    manager = RecursionManager(
        RecursionGuard(2), lambda *args, **kwargs: ("answer", None),
        budget=budget, token_tracker=tracker,
    )
    tools = ToolInterface()
    tools.register("llm_query", manager.run_subtask)
    runtime = RuntimeEngine(StateStore(), tools, budget=budget, token_tracker=tracker, before_step=manager.begin_step)
    result = runtime.execute("result = llm_query('question')")
    assert result.ok
    assert budget.used == budget.uncertain > 0 and budget.reserved == 0
    assert tracker.counts == []


def test_system_budget_is_not_silently_reset_between_runs():
    budget = BudgetManager(100000)
    outputs = iter([
        {"action": "execute", "code": "result='a'"},
        {"action": "finalize_variable", "name": "result"},
        {"action": "execute", "code": "result='b'"},
        {"action": "finalize_variable", "name": "result"},
    ])
    generator = CodeGenerator(chat=lambda *args, **kwargs: (json.dumps(next(outputs)), 10), budget=budget)
    controller = RootController(RuntimeEngine(StateStore(), ToolInterface(), timeout_seconds=None, budget=budget), codegen=generator, require_evidence=False)
    first = controller.run_until_done(MountedContext("context"), "first")
    second = controller.run_until_done(MountedContext("context"), "second")
    assert first.require_answer() == "a" and second.require_answer() == "b"
    assert budget.used == 40
    assert second.token_usage["run_charged"] == 20


def test_groq_uses_one_retry_owner_json_mode_timeout_and_unknown_usage(monkeypatch):
    from shared.groq_client import make_groq_chat, make_groq_llm
    constructors, requests = [], []

    class Client:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"execute","code":"result=1"}'))],
                usage=None,
            )

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=Client))
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    chat = make_groq_chat(timeout_seconds=9)
    _, usage = chat([{"role": "user", "content": "query"}], max_tokens=10, timeout_seconds=2)
    assert usage is None
    assert constructors[0]["max_retries"] == 0
    assert requests[0]["timeout"] == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    make_groq_llm(timeout_seconds=9)("query", max_tokens=10)
    assert "response_format" not in requests[-1]
    assert requests[-1]["timeout"] == 9


def test_replacement_codegen_inherits_system_budget_instead_of_bypassing_it():
    budget = BudgetManager(100000)
    runtime = RuntimeEngine(StateStore(), ToolInterface(), timeout_seconds=None, budget=budget)
    controller = RootController(runtime, require_evidence=False)
    actions = iter([
        {"action": "execute", "code": "result='answer'"},
        {"action": "finalize_variable", "name": "result"},
    ])
    controller.codegen = CodeGenerator(chat=lambda *args, **kwargs: (json.dumps(next(actions)), 7))
    outcome = controller.run_until_done(MountedContext("document"), "query")
    assert outcome.ok
    assert controller.codegen.budget is budget
    assert budget.used == 14 and budget.reserved == 0


def test_reservations_cannot_be_double_settled_or_used_with_another_budget():
    budget, other = BudgetManager(100), BudgetManager(100)
    first, second = budget.reserve(20), budget.reserve(20)
    foreign = other.reserve(20)
    budget.settle(first, 5)
    with pytest.raises(ValueError, match="Unknown reservation"):
        budget.settle(first, 5)
    with pytest.raises(ValueError, match="Unknown reservation"):
        budget.settle(foreign, 5)
    assert budget.reserved == 20 and budget.used == 5
    budget.settle(second, 5)
    other.settle(foreign, 5)
    assert budget.reserved == other.reserved == 0


def test_accounting_tracker_failure_does_not_leak_a_known_usage_reservation():
    class BrokenTracker:
        def record(self, tokens):
            raise MemoryError()

    budget = BudgetManager(100)
    with pytest.raises(MemoryError):
        call_model(
            lambda *args, **kwargs: ("answer", 5), "task", budget=budget,
            tracker=BrokenTracker(), max_tokens=20, token_counter=lambda _: 10,
        )
    assert budget.used == 5 and budget.reserved == 0


def test_unknown_usage_callback_failure_does_not_leak_a_reservation():
    budget = BudgetManager(100)

    def broken():
        raise MemoryError()

    with pytest.raises(MemoryError):
        call_model(
            lambda *args, **kwargs: ("answer", None), "task", budget=budget,
            max_tokens=20, token_counter=lambda _: 10, on_unknown=broken,
        )
    assert budget.used == budget.uncertain == 30 and budget.reserved == 0
