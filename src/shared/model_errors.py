"""Safe, typed failures shared by model callers and the controller."""


class ModelError(RuntimeError):
    code = "provider_error"
    retryable = False

    def __init__(self, message="Model request failed", *, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceededError(ModelError):
    code = "budget_exhausted"


class HistoryBudgetError(ModelError):
    code = "context_limit"


class DeadlineExceededError(ModelError):
    code = "deadline_exceeded"


class AuthenticationError(ModelError):
    code = "authentication_failed"


class ProviderRequestError(ModelError):
    code = "provider_request_rejected"


class ModelNotConfiguredError(ModelError):
    code = "model_not_configured"


class InvalidTaskError(ModelError):
    code = "invalid_task"


class RateLimitError(ModelError):
    code = "rate_limited"
    retryable = True


class ProviderTimeoutError(ModelError):
    code = "provider_timeout"
    retryable = True


class ProviderUnavailableError(ModelError):
    code = "provider_unavailable"
    retryable = True


class InvalidUsageError(ModelError):
    code = "invalid_usage"


class ProtocolError(ModelError):
    code = "invalid_action"


class EvidenceError(ModelError):
    code = "invalid_evidence"


class VerificationError(ModelError):
    code = "verification_failed"


class SourceUnavailableError(ModelError):
    code = "source_unavailable"


class SubcallLimitError(ModelError):
    code = "subcall_limit"


class StepLimitError(ModelError):
    code = "step_limit"
