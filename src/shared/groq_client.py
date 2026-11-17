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
    timeout_seconds: float = 30,
) -> "LLMCallable":
    """Single-turn LLMCallable — compatible with CodeGenerator / RecursionManager."""
    from groq import Groq  # type: ignore[import]

    request_timeout = timeout_seconds
    client = Groq(api_key=groq_api_key(), max_retries=0, timeout=request_timeout)

    def call(prompt: str, *, max_tokens: int | None = None, timeout_seconds: float | None = None) -> tuple[str, int | None]:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
            timeout=timeout_seconds or request_timeout,
        )
        msg = completion.choices[0].message
        tokens = completion.usage.total_tokens if completion.usage else None
        return ((msg.content or "").strip(), tokens)

    return call


def make_groq_chat(
    model: str = _DEFAULT_ROOT_MODEL,
    *,
    temperature: float = 0.2,
    timeout_seconds: float = 30,
    json_mode: bool = True,
) -> "ChatCallable":
    """Multi-turn ChatCallable — accepts list[dict] messages, returns str.

    Used by CodeGenerator.generate_step() for the full RLM REPL loop.
    """
    from groq import Groq  # type: ignore[import]

    request_timeout = timeout_seconds
    client = Groq(api_key=groq_api_key(), max_retries=0, timeout=request_timeout)

    def chat(
        messages: list[dict[str, str]], *, max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[str, int | None]:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            timeout=timeout_seconds or request_timeout,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        )
        msg = completion.choices[0].message
        tokens = completion.usage.total_tokens if completion.usage else None
        return ((msg.content or "").strip(), tokens)

    return chat


def optional_groq_llm() -> Callable[..., tuple[str, int | None]] | None:
    """If GROQ_API_KEY is set, return Groq caller; else None."""
    if not os.environ.get("GROQ_API_KEY", "").strip():
        return None
    return make_groq_llm()
