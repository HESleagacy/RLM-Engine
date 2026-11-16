"""One bounded JSON action per response; no interpretation of prose/code markers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import keyword
import re

from shared.model_errors import ProtocolError

PHASES = ("inspect", "retrieve", "analyze", "synthesize")


@dataclass(frozen=True)
class EvidenceRef:
    document_hash: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Action:
    action: str
    code: str | None = None
    text: str | None = None
    name: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    phase: str | None = None

    def to_dict(self) -> dict:
        if self.action == "execute":
            value = {"action": self.action, "code": self.code}
            if self.phase is not None:
                value["phase"] = self.phase
            return value
        return {
            "action": self.action,
            "text" if self.action == "finalize" else "name":
                self.text if self.action == "finalize" else self.name,
            "evidence": [asdict(ref) for ref in self.evidence],
        }


def _identifier(value) -> bool:
    return (
        isinstance(value, str) and len(value) <= 128 and value.isidentifier()
        and not keyword.iskeyword(value) and not value.startswith("_")
    )


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"Duplicate JSON field: {key[:64]}")
        value[key] = item
    return value


def parse_action(raw: str, *, max_response_bytes: int = 65_536) -> Action:
    if not isinstance(raw, str) or len(raw) > max_response_bytes or len(raw.encode("utf-8")) > max_response_bytes:
        raise ProtocolError("Response must be a bounded JSON string")
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("Non-finite JSON number")),
        )
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("Return exactly one JSON action, without markdown or commentary") from exc
    return validate_action(value)


def validate_action(value: dict | Action) -> Action:
    if isinstance(value, Action):
        try:
            value = value.to_dict()
        except (TypeError, AttributeError) as exc:
            raise ProtocolError("Action contains invalid typed fields") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Action must be one JSON object")
    kind = value.get("action")
    if kind == "execute":
        if set(value) - {"action", "code", "phase"}:
            raise ProtocolError("Execution cannot contain finalization or unknown fields")
        code, phase = value.get("code"), value.get("phase")
        if not isinstance(code, str) or not code.strip() or len(code) > 32_768:
            raise ProtocolError("Execution requires nonempty code of at most 32768 characters")
        if phase is not None and phase not in PHASES:
            raise ProtocolError("Unknown execution phase")
        return Action(kind, code=code, phase=phase)
    if kind not in ("finalize", "finalize_variable"):
        raise ProtocolError("Unknown action; use execute, finalize, or finalize_variable")
    field = "text" if kind == "finalize" else "name"
    if set(value) - {"action", field, "evidence"}:
        raise ProtocolError("Finalization contains mixed actions or unknown fields")
    answer = value.get(field)
    if field == "text" and (not isinstance(answer, str) or len(answer) > 16_384):
        raise ProtocolError("Final text must be a string of at most 16384 characters")
    if field == "name" and not _identifier(answer):
        raise ProtocolError("Final variable must be a public Python identifier")
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) > 32:
        raise ProtocolError("Evidence must be a list of at most 32 references")
    refs = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"document_hash", "start_char", "end_char"}:
            raise ProtocolError("Evidence requires document_hash, start_char, and end_char")
        digest, start, end = item["document_hash"], item["start_char"], item["end_char"]
        if (
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(start) is not int or type(end) is not int or not 0 <= start < end
        ):
            raise ProtocolError("Invalid evidence identity or character range")
        refs.append(EvidenceRef(digest, start, end))
    return Action(kind, text=answer if field == "text" else None,
                  name=answer if field == "name" else None, evidence=tuple(refs))


def parse_legacy_action(raw: str) -> Action:
    """Explicit compatibility only; markers inside code are never finalization."""
    text = raw.strip()
    variable = re.fullmatch(r"FINAL_VAR\(([A-Za-z_]\w*)\)", text)
    if variable:
        return validate_action({"action": "finalize_variable", "name": variable[1]})
    if text.startswith("FINAL(") and text.endswith(")"):
        return validate_action({"action": "finalize", "text": text[6:-1]})
    code = re.fullmatch(r"```(?:repl|python)\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if code and "```" not in code[1]:
        return validate_action({"action": "execute", "code": code[1]})
    raise ProtocolError("Invalid legacy action; mixed blocks and prose are not supported")
