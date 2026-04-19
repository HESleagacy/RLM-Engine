"""Groq chat client — API key from environment only (never hardcode)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from shared.types import LLMCallable


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in the environment (e.g. export GROQ_API_KEY=...); do not commit secrets."
        )
    return key


def make_groq_llm(
    model: str = "llama-3.1-8b-instant",
    *,
    temperature: float = 0.2,
) -> "LLMCallable":
    """Return a callable compatible with CodeGenerator / sub-LLM hooks."""
    from groq import Groq

    client = Groq(api_key=groq_api_key())

    def call(prompt: str, *, max_tokens: int | None = None) -> str:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )
        msg = completion.choices[0].message
        return (msg.content or "").strip()

    return call


def optional_groq_llm() -> Callable[[str], str] | None:
    """If GROQ_API_KEY is set, return Groq caller; else None."""
    if not os.environ.get("GROQ_API_KEY", "").strip():
        return None
    return make_groq_llm()
