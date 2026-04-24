"""Groq client factories — single-turn LLM and multi-turn chat.

API key is read from the environment only (never hardcoded).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from shared.types import ChatCallable, LLMCallable

_DEFAULT_ROOT_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_SUB_MODEL = "llama-3.1-8b-instant"


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in the environment "
            "(e.g. export GROQ_API_KEY=...); do not commit secrets."
        )
    return key


def make_groq_llm(
    model: str = _DEFAULT_SUB_MODEL,
    *,
    temperature: float = 0.2,
) -> "LLMCallable":
    """Single-turn LLMCallable — compatible with CodeGenerator / RecursionManager."""
    from groq import Groq  # type: ignore[import]

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


def make_groq_chat(
    model: str = _DEFAULT_ROOT_MODEL,
    *,
    temperature: float = 0.2,
) -> "ChatCallable":
    """Multi-turn ChatCallable — accepts list[dict] messages, returns str.

    Used by CodeGenerator.generate_step() for the full RLM REPL loop.
    """
    from groq import Groq  # type: ignore[import]

    client = Groq(api_key=groq_api_key())

    def chat(
        messages: list[dict[str, str]], *, max_tokens: int | None = None
    ) -> str:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens or 4096,
        )
        msg = completion.choices[0].message
        return (msg.content or "").strip()

    return chat


def optional_groq_llm() -> Callable[[str], str] | None:
    """If GROQ_API_KEY is set, return Groq caller; else None."""
    if not os.environ.get("GROQ_API_KEY", "").strip():
        return None
    return make_groq_llm()
