"""One retry owner, preflight reservations, explicit uncertain usage, and deadlines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time

from shared.model_errors import (
    AuthenticationError, BudgetExceededError, DeadlineExceededError,
    InvalidUsageError, ModelError, ProviderRequestError, ProviderTimeoutError,
    ProviderUnavailableError, RateLimitError,
)
from shared.resource_errors import resource_error


def estimate_input_tokens(payload) -> int:
    """Conservative byte-based estimate, not an exact provider tokenizer."""
    messages = payload if isinstance(payload, list) else [{"content": payload}]
    return 128 + sum(len(message["content"].encode("utf-8")) + 32 for message in messages)


def classify_provider_error(exc: Exception) -> ModelError:
    if isinstance(exc, ModelError):
        return exc
    status = getattr(exc, "status_code", None)
    if type(status) is not int:
        status = None
    if status in (401, 403):
        return AuthenticationError("Provider authentication or permission failed")
    if status == 429:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        try:
            retry_after = max(0.0, float(headers.get("retry-after", 0)))
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(headers.get("retry-after", ""))
                retry_after = max(0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                retry_after = None
        if retry_after is not None and not math.isfinite(retry_after):
            retry_after = None
        return RateLimitError("Provider rate limit reached", retry_after=retry_after)
    if status is not None and 500 <= status <= 599:
        return ProviderUnavailableError("Provider is temporarily unavailable")
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return ProviderTimeoutError("Provider request timed out")
    if isinstance(exc, ConnectionError) or "connection" in type(exc).__name__.lower():
        return ProviderUnavailableError("Provider connection failed")
    if status is not None and 400 <= status <= 499:
        return ProviderRequestError("Provider rejected the request")
    return ModelError("Provider request failed")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.5
    max_delay_seconds: float = 30
    request_timeout_seconds: float = 30

    def __post_init__(self):
        if (
            type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 5
            or not all(math.isfinite(value) for value in (
                self.backoff_seconds, self.max_delay_seconds, self.request_timeout_seconds
            )) or self.backoff_seconds < 0
            or self.max_delay_seconds < 0 or self.request_timeout_seconds <= 0
        ):
            raise ValueError("Invalid retry policy")


def call_model(
    function, payload, *, budget=None, tracker=None, max_tokens=4096,
    deadline=None, policy=None, token_counter=estimate_input_tokens,
    on_attempt=None, on_unknown=None, clock=time.monotonic, sleep=time.sleep,
):
    policy = policy or RetryPolicy()
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("Output-token limit must be a positive integer")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("Deadline must be finite")
    input_tokens = token_counter(payload)
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("Token counter must return a non-negative integer")
    for attempt in range(policy.max_attempts):
        remaining_time = None if deadline is None else deadline - clock()
        if remaining_time is not None and remaining_time <= 0:
            raise DeadlineExceededError("Run deadline exceeded")
        output_tokens = max_tokens
        if budget is not None:
            output_tokens = min(output_tokens, budget.remaining() - input_tokens)
        if output_tokens <= 0:
            raise BudgetExceededError("Insufficient budget for input and output")
        reservation = None
        if budget is not None:
            reservation = budget.reserve(input_tokens + output_tokens)
        kwargs = {
            "max_tokens": output_tokens,
            "timeout_seconds": policy.request_timeout_seconds if remaining_time is None else min(policy.request_timeout_seconds, remaining_time),
        }
        try:
            if on_attempt:
                on_attempt()
            response, tokens = function(payload, **kwargs)
        except BaseException as exc:
            # Unknown billed usage is charged conservatively; retries are not free.
            if budget is not None:
                try:
                    budget.settle(reservation, None)
                except BudgetExceededError:
                    if resource_error(exc) is None:
                        raise
            if on_unknown:
                on_unknown()
            if not isinstance(exc, Exception) or resource_error(exc) is not None:
                raise
            failure = classify_provider_error(exc)
            if not failure.retryable or attempt + 1 == policy.max_attempts:
                if failure is exc:
                    raise
                raise failure from exc
            delay = max(policy.backoff_seconds * (2 ** attempt), failure.retry_after or 0)
            if delay > policy.max_delay_seconds:
                if failure is exc:
                    raise
                raise failure from exc
            if deadline is not None and clock() + delay >= deadline:
                raise DeadlineExceededError("Insufficient time for another provider attempt") from exc
            sleep(delay)
            continue
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            if budget is not None:
                budget.settle(reservation, None)
            if on_unknown:
                on_unknown()
            raise InvalidUsageError("Provider returned invalid token usage")
        settlement_error = None
        if budget is not None:
            try:
                budget.settle(reservation, tokens)
            except BudgetExceededError as exc:
                settlement_error = exc
        if tracker is not None and tokens is not None:
            tracker.record(tokens)
        if tokens is None and on_unknown:
            on_unknown()
        if settlement_error is not None:
            raise settlement_error
        if deadline is not None and clock() >= deadline:
            raise DeadlineExceededError("Provider response arrived after the run deadline")
        return response, tokens
    raise AssertionError("Unreachable retry state")
