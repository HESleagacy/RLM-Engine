"""Host-owned source-read ledger; provenance checks are not truth verification."""

from dataclasses import asdict
import hashlib
import time

from layer2_controller.protocol import EvidenceRef
from shared.model_errors import DeadlineExceededError, EvidenceError, ModelError, SourceUnavailableError
from shared.resource_errors import resource_error


class EvidenceLedger:
    def __init__(self, context, *, max_span_chars=4096, max_entries=64, deadline=None, clock=time.monotonic):
        self.context = context
        self.max_span_chars = max_span_chars
        self.max_entries = max_entries
        digest = hashlib.sha256()
        try:
            for start in range(0, self.length, 65_536):
                if deadline is not None and clock() >= deadline:
                    raise DeadlineExceededError("Run deadline exceeded while identifying the source")
                digest.update(self._slice(start, start + 65_536).encode("utf-8"))
        except ModelError:
            raise
        except (OSError, RuntimeError) as exc:
            if resource_error(exc):
                raise
            raise SourceUnavailableError("Mounted source is unavailable or changed") from exc
        self.document_hash = digest.hexdigest()
        self.records = {}
        self.step_records = {}

    def begin_step(self):
        self.step_records = {}

    @property
    def length(self):
        try:
            return len(self.context.text)
        except (OSError, RuntimeError) as exc:
            if resource_error(exc):
                raise
            raise SourceUnavailableError("Mounted source is unavailable or changed") from exc

    def _slice(self, start, end):
        try:
            return self.context.text[start:end]
        except (OSError, RuntimeError) as exc:
            if resource_error(exc):
                raise
            raise SourceUnavailableError("Mounted source is unavailable or changed") from exc

    def read(self, start_char: int, end_char: int) -> dict:
        if (
            type(start_char) is not int or type(end_char) is not int
            or not 0 <= start_char < end_char <= self.length
            or end_char - start_char > self.max_span_chars
        ):
            raise EvidenceError(f"Read requires a valid source span of at most {self.max_span_chars} characters")
        key = (start_char, end_char)
        if key not in self.records and key not in self.step_records:
            if len(set(self.records) | set(self.step_records)) >= self.max_entries:
                raise EvidenceError("Evidence limit reached; reuse existing source references")
        record = {
            **asdict(EvidenceRef(self.document_hash, start_char, end_char)),
            "text": self._slice(start_char, end_char),
        }
        self.step_records[key] = record
        return dict(record)

    def issued(self) -> tuple[dict, ...]:
        return tuple(self.step_records.values())

    def ingest(self, records):
        for record in records:
            self.records[(record["start_char"], record["end_char"])] = record

    def catalog(self) -> list[dict]:
        return [
            {key: record[key] for key in ("document_hash", "start_char", "end_char")}
            for record in self.records.values()
        ]

    def validate(self, references) -> tuple[dict, ...]:
        verified = []
        for ref in references:
            if ref.document_hash != self.document_hash:
                raise EvidenceError("Evidence belongs to a different document")
            record = self.records.get((ref.start_char, ref.end_char))
            if record is None:
                raise EvidenceError("Evidence span was not read through read_context")
            # Reject source changes, including after a cache hit or earlier read.
            if self._slice(ref.start_char, ref.end_char) != record["text"]:
                raise EvidenceError("Source evidence changed after reading")
            verified.append(record)
        return tuple(verified)
