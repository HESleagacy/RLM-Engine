"""Code generation — produce Python for the runtime, or drive the RLM REPL loop.

Gap 6 fix: CodeGenerator.generate_step() uses the paper's system prompt
(Appendix D.1) and supports multi-turn chat via ChatCallable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.types import ChatCallable, LLMCallable

# ── Legacy single-turn prefix (backward-compatible with tests) ────────────────

_LLM_SYSTEM_PREFIX = """You are a code emitter for a Python exec() sandbox.
Output ONLY valid Python. No markdown fences unless they wrap pure Python.
You MUST assign the final answer to a variable named `result` (string, number, or object).
Do not print explanations. One short block is enough.

Task:
"""

# ── Gap 6: Paper's system prompt (Appendix D.1), adapted for the RLM loop ─────

_RLM_SYSTEM_PROMPT = """\
You are tasked with answering a query with associated context. You can access, transform, and \
analyze this context interactively in a REPL environment that can recursively query sub-LLMs, \
which you are strongly encouraged to use. You will be queried iteratively until you provide a \
final answer.

Your context has {context_total_length} total characters and is stored in the 'context' variable.

The REPL environment provides:
1. A 'context' variable containing the full context for your query. Always inspect it before answering.
2. A 'llm_query(prompt: str) -> str' function to query a sub-LLM inside the REPL. The sub-LLM \
   can handle around 200K characters per call.
3. Full use of 'print()' to observe intermediate results between rounds.

You will only see truncated REPL output, so use llm_query() for semantic analysis. Use variables \
as buffers to build up your final answer.

IMPORTANT: Batch information into llm_query calls — aim for ~200K characters per call. Making one \
call per line is extremely wasteful. Always chunk your context into a reasonable number of pieces.

Strategy: probe context → decide chunking → llm_query per chunk → aggregate → finalize.

Example — chunk and aggregate:
```repl
chunk_size = max(1, len(context) // 5)
answers = []
for i in range(5):
    chunk = context[i * chunk_size : (i + 1) * chunk_size]
    ans = llm_query(f"Answer this: {{query}}\\nContext chunk:\\n{{chunk}}")
    answers.append(ans)
    print(f"chunk {{i}}: {{ans[:150]}}")
final = llm_query(f"Combine these partial answers for '{{query}}':\\n" + "\\n".join(answers))
print(final)
```

Example — probe then filter:
```repl
print(context[:2000])          # inspect head
import re
hits = [ln for ln in context.splitlines() if re.search(r"keyword", ln, re.I)]
print(f"found {{len(hits)}} matching lines")
answer = llm_query(f"From these lines, answer: {{query}}\\n" + "\\n".join(hits[:200]))
print(answer)
```

IMPORTANT — finishing: when you have your final answer, write ONE of the following OUTSIDE any \
code block. Do NOT wrap it in backticks:

    FINAL(your complete answer here)
    FINAL_VAR(variable_name)      ← returns a variable stored in the REPL

Think step by step, execute immediately, and never just say what you will do — do it.\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_fenced_code(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:python)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_repl_block(text: str) -> str | None:
    """Return the first ```repl or ```python block, or None."""
    m = re.search(r"```(?:repl|python)\s*\\n([\\s\\S]*?)\\n```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _coerce_to_executable(source: str) -> str:
    """If the model returned prose or invalid syntax, wrap it as a string result."""
    body = _extract_fenced_code(source)
    try:
        compile(body, "<generated>", "exec")
    except SyntaxError:
        return f"result = {body!r}\\n"
    if "result" not in body:
        return f"{body}\\nresult = None\\n"
    return body if body.endswith("\\n") else body + "\\n"


# ── StepResult ────────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Output of one generate_step() call — either a code block or a FINAL signal."""

    raw: str
    code: str | None = None       # extracted ```repl block (if any)
    is_final: bool = False
    final_text: str | None = None  # from FINAL(...)
    final_var: str | None = None   # from FINAL_VAR(...)


def _parse_step(raw: str) -> StepResult:
    """Parse raw LLM response text into a StepResult."""
    # FINAL_VAR takes priority (more specific pattern)
    m_var = re.search(r"FINAL_VAR\(([^)]+)\)", raw)
    if m_var:
        return StepResult(raw=raw, is_final=True, final_var=m_var.group(1).strip())

    # FINAL(answer) — may span multiple lines
    m_fin = re.search(r"FINAL\((.+?)\)", raw, re.DOTALL)
    if m_fin:
        return StepResult(raw=raw, is_final=True, final_text=m_fin.group(1).strip())

    # Fallback for LLMs that forget parentheses: "FINAL prompt answer here" or "FINAL: answer"
    m_fin_loose = re.search(r"(?:^|\n)FINAL[:\s]+(.+)", raw)
    if m_fin_loose:
        return StepResult(raw=raw, is_final=True, final_text=m_fin_loose.group(1).strip())

    # Regular repl code block
    code = _extract_repl_block(raw)
    return StepResult(raw=raw, code=code)


# ── CodeGenerator ─────────────────────────────────────────────────────────────


class CodeGenerator:
    def __init__(
        self,
        llm: "LLMCallable | None" = None,
        chat: "ChatCallable | None" = None,
    ) -> None:
        self._llm = llm
        self._chat = chat  # preferred for multi-turn REPL loop

    # ── Legacy single-turn method (backward-compatible, used by tests) ─────────

    def generate(self, instruction: str, context_excerpt: str) -> str:
        """Single-turn fallback — produces executable Python for run_round()."""
        if self._llm is not None:
            prompt = (
                f"{_LLM_SYSTEM_PREFIX}{instruction}\n\n"
                f"--- Context (mounted prompt P) ---\n{context_excerpt}\n"
            )
            raw = self._llm(prompt).strip()
            return _coerce_to_executable(raw)
        # Deterministic fallback for tests (no LLM needed)
        return f"result = {context_excerpt!r}\n"

    # ── Gap 6: Multi-turn REPL step for run_until_done() ──────────────────────

    def generate_step(
        self,
        query: str,
        context_total_length: int,
        history: list[dict[str, str]],
    ) -> StepResult:
        """
        Drive one round of the RLM REPL loop.

        `history` — list of {"code": str, "output": str} dicts from prior rounds.
        Returns a StepResult with either a repl code block or a FINAL signal.
        """
        system = _RLM_SYSTEM_PROMPT.format(context_total_length=context_total_length)

        if self._chat is not None:
            # Full multi-turn chat — preferred path
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]
            for h in history:
                messages.append(
                    {"role": "assistant", "content": f"```repl\n{h['code']}\n```"}
                )
                messages.append(
                    {"role": "user", "content": f"REPL output:\n{h['output']}"}
                )
            raw = self._chat(messages)

        elif self._llm is not None:
            # Flatten history into a single prompt (degraded path)
            flat = f"{system}\n\nQuery: {query}"
            for h in history:
                flat += f"\n\n```repl\n{h['code']}\n```\nREPL output:\n{h['output']}"
            raw = self._llm(flat)

        else:
            # Deterministic test fallback — immediately signal done
            return StepResult(
                raw="FINAL(test_answer)", is_final=True, final_text="test_answer"
            )

        return _parse_step(raw)
