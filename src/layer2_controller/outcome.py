"""Answers and failures are different outcomes, never interchangeable strings."""

from dataclasses import asdict, dataclass

from layer2_controller.protocol import EvidenceRef


@dataclass(frozen=True)
class StepObservation:
    round: int
    ok: bool
    stdout_preview: str = ""
    stdout_truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None
    state_summary: dict | None = None
    evidence: tuple[dict, ...] = ()


@dataclass(frozen=True)
class RunOutcome:
    status: str
    stop_reason: str
    answer: str | None = None
    partial_candidate: str | None = None
    error_code: str | None = None
    message: str | None = None
    observations: tuple[StepObservation, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    document_hash: str | None = None
    verification: str = "unverified"
    model_attempts: int = 0
    history_compactions: int = 0
    usage_uncertain: bool = False
    token_usage: dict | None = None
    dropped_observations: int = 0
    protocol: str = "json"

    def __post_init__(self):
        if self.status not in ("completed", "failed", "incomplete", "cancelled"):
            raise ValueError("Run outcomes require a terminal status")
        if self.status == "completed":
            if not isinstance(self.answer, str) or self.error_code is not None:
                raise ValueError("Completed outcomes require an answer and cannot contain an error")
        elif self.answer is not None or self.verification != "unverified":
            raise ValueError("Noncompleted outcomes cannot contain a final answer or verification claim")
        if self.verification not in ("unverified", "provenance_validated", "externally_verified"):
            raise ValueError("Unknown verification scope")
        if self.verification == "provenance_validated" and not self.evidence:
            raise ValueError("Provenance validation requires evidence")

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict:
        return asdict(self)

    def require_answer(self) -> str:
        if not self.ok or self.answer is None:
            raise RunFailedError(self)
        return self.answer


class RunFailedError(RuntimeError):
    def __init__(self, outcome: RunOutcome):
        self.outcome = outcome
        self.code = outcome.error_code or outcome.stop_reason
        super().__init__(outcome.message or f"Run stopped: {outcome.stop_reason}")

    @property
    def status_code(self) -> int:
        if self.code in ("disk_full", "out_of_memory"):
            return 503
        if self.code in ("deadline_exceeded", "provider_timeout"):
            return 504
        if self.code == "rate_limited":
            return 429
        return 422 if self.outcome.status in ("incomplete", "cancelled") else 502
