"""Code generation — produce Python for the runtime."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.types import LLMCallable

_LLM_SYSTEM_PREFIX = """You are a code emitter for a Python exec() sandbox.
Output ONLY valid Python. No markdown fences unless they wrap pure Python.
You MUST assign the final answer to a variable named `result` (string, number, or object).
Do not print explanations. One short block is enough.

Task:
"""


def _extract_fenced_code(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:python)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _coerce_to_executable(source: str) -> str:
    """If the model returned prose or invalid syntax, assign it to result as a string."""
    body = _extract_fenced_code(source)
    try:
        compile(body, "<generated>", "exec")
    except SyntaxError:
        return f"result = {body!r}\n"
    if "result" not in body:
        return f"{body}\nresult = None\n"
    return body if body.endswith("\n") else body + "\n"


class CodeGenerator:
    def __init__(self, llm: "LLMCallable | None" = None) -> None:
        self._llm = llm

    def generate(self, instruction: str, context_excerpt: str) -> str:
        if self._llm is not None:
            prompt = f"{_LLM_SYSTEM_PREFIX}{instruction}\n\n--- Context (mounted prompt P) ---\n{context_excerpt}\n"
            raw = self._llm(prompt).strip()
            return _coerce_to_executable(raw)
        # Deterministic fallback for tests
        return f"result = {context_excerpt!r}\n"
