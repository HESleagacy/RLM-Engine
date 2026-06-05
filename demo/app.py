"""Consistency Challenge Demo — RAG vs RLM side-by-side story generation.

Launch:
    pip install gradio
    PYTHONPATH=src GROQ_API_KEY=... python demo/app.py

The UI lets you:
  1. Enter or generate a fact-scattered document
  2. Run both RAG and RLM pipelines
  3. Compare stories side-by-side with fact-accuracy scores
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on the path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import gradio as gr

from layer8_evaluation.benchmarks.consistency import (
    ConsistencyResult,
    FactSet,
    build_query,
    build_scattered_document,
)


# ── LLM helpers ──────────────────────────────────────────────────────────────


def _get_llm():
    """Return a simple string→string LLM caller (Groq)."""
    try:
        from shared.groq_client import make_groq_llm
        llm = make_groq_llm()
        def call(prompt: str) -> str:
            text, _ = llm(prompt, max_tokens=2048)
            return text
        return call
    except Exception as e:
        return None


# ── RAG Pipeline ─────────────────────────────────────────────────────────────


def run_rag(document: str, query: str, llm_fn) -> str:
    """Simple RAG: BM25 retrieval of top-5 chunks → single LLM call."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return "[ERROR] rank_bm25 not installed. Run: pip install rank_bm25"

    # Split into paragraphs as chunks
    chunks = [p.strip() for p in document.split("\n\n") if p.strip()]
    if not chunks:
        return "[ERROR] No content in document."

    # BM25 retrieval — top 5 chunks
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    top_chunks = bm25.get_top_n(query_tokens, chunks, n=5)

    retrieved_context = "\n\n".join(top_chunks)

    prompt = (
        f"You are a creative writer. Based ONLY on the following retrieved context, "
        f"write a short story (3-5 paragraphs).\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"Task:\n{query}"
    )

    return llm_fn(prompt)


# ── RLM Pipeline ─────────────────────────────────────────────────────────────


def run_rlm(document: str, query: str, llm_fn) -> str:
    """RLM: mount full document, let the LLM programmatically explore it."""
    from layer1_input.context_repr import MountedContext
    from layer1_input.raw_loader import normalize
    from layer2_controller.code_generator import CodeGenerator
    from layer2_controller.controller import RootController
    from layer3_execution.runtime_engine import RuntimeEngine
    from layer3_execution.state_store import StateStore
    from layer3_execution.tool_interface import ToolInterface
    from layer4_recursion.recursion_manager import RecursionManager
    from layer5_context_access import by_keyword, by_regex, fixed_windows, peek_head, peek_tail
    from layer7_control.budget_manager import BudgetManager
    from layer7_control.recursion_guard import RecursionGuard
    from layer7_control.step_limiter import StepLimiter
    from shared.groq_client import make_groq_chat, make_groq_llm

    # Build the system
    steps = StepLimiter(max_steps=50)
    budget = BudgetManager(limit=50_000)
    guard = RecursionGuard(max_depth=3)

    state = StateStore()
    tools = ToolInterface()
    runtime = RuntimeEngine(state, tools, step_limiter=steps, strict_sandbox=True)

    # Register context access tools
    def wrap_tool(fn):
        def wrapper(*args, **kwargs):
            ctx_text = state.get("context", "")
            return fn(MountedContext(text=ctx_text), *args, **kwargs)
        return wrapper

    tools.register("peek_head", wrap_tool(peek_head))
    tools.register("peek_tail", wrap_tool(peek_tail))
    tools.register("by_keyword", wrap_tool(by_keyword))
    tools.register("by_regex", wrap_tool(by_regex))
    tools.register("chunker", wrap_tool(fixed_windows))

    # Wire sub-LLM
    sub_llm = make_groq_llm(model="llama-3.1-8b-instant")
    rec_manager = RecursionManager(guard=guard, llm=sub_llm, budget=budget)
    tools.register("llm_query", rec_manager.run_subtask)

    # Wire root LLM
    root_llm = make_groq_llm(model="llama-3.3-70b-versatile")
    root_chat = make_groq_chat(model="llama-3.3-70b-versatile")
    codegen = CodeGenerator(llm=root_llm, chat=root_chat, budget=budget)

    controller = RootController(
        runtime, codegen=codegen, max_rounds=10, stdout_truncation=3000
    )

    ctx = MountedContext(text=normalize(document))
    answer = controller.run_until_done(ctx, query)
    return answer.text


# ── Scoring ──────────────────────────────────────────────────────────────────


def score_story(story: str, facts_dict: dict[str, str]) -> ConsistencyResult:
    fact_set = FactSet(facts=facts_dict)
    return ConsistencyResult.from_story(story, fact_set)


def format_result(result: ConsistencyResult) -> str:
    lines = [
        f"**Accuracy: {result.accuracy:.0%}** ({len(result.present)}/{len(result.present) + len(result.missing)} facts used)",
        "",
    ]
    if result.present:
        lines.append("✅ **Present:**")
        for k in result.present:
            lines.append(f"  - {k}: {result.fact_set.facts[k]}")
    if result.missing:
        lines.append("")
        lines.append("❌ **Missing:**")
        for k in result.missing:
            lines.append(f"  - {k}: {result.fact_set.facts[k]}")
    return "\n".join(lines)


# ── Gradio UI ────────────────────────────────────────────────────────────────


def generate_document(seed: int = 42) -> tuple[str, str]:
    """Generate a fact-scattered document and return (document, facts_display)."""
    fact_set, doc = build_scattered_document(seed=seed)
    facts_display = "\n".join(fact_set.as_list())
    return doc, facts_display


def run_comparison(document: str, facts_text: str):
    """Run both RAG and RLM, return stories + scores."""
    llm_fn = _get_llm()
    if llm_fn is None:
        return (
            "[ERROR] GROQ_API_KEY not set. Export it and restart.",
            "[ERROR] GROQ_API_KEY not set. Export it and restart.",
            "N/A",
            "N/A",
        )

    # Parse facts from display text
    facts_dict = {}
    for line in facts_text.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            facts_dict[k.strip()] = v.strip()

    if not facts_dict:
        return "No facts found.", "No facts found.", "N/A", "N/A"

    fact_set = FactSet(facts=facts_dict)
    query = build_query(fact_set)

    # Run RAG
    try:
        rag_story = run_rag(document, query, llm_fn)
    except Exception as e:
        rag_story = f"[RAG ERROR] {e}"

    # Run RLM
    try:
        rlm_story = run_rlm(document, query, llm_fn)
    except Exception as e:
        rlm_story = f"[RLM ERROR] {e}"

    # Score both
    rag_result = score_story(rag_story, facts_dict)
    rlm_result = score_story(rlm_story, facts_dict)

    return rag_story, rlm_story, format_result(rag_result), format_result(rlm_result)


def build_ui():
    with gr.Blocks(
        title="RLM vs RAG — Consistency Challenge",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🧠 RLM vs RAG — Consistency Challenge\n"
            "Generate a fact-scattered document, then compare how **RAG** (retrieval-only) "
            "and **RLM** (recursive reasoning) handle story generation with all facts."
        )

        with gr.Row():
            with gr.Column(scale=2):
                document_box = gr.Textbox(
                    label="📄 Document (facts scattered in filler text)",
                    lines=15,
                    placeholder="Click 'Generate Document' or paste your own...",
                )
            with gr.Column(scale=1):
                facts_box = gr.Textbox(
                    label="📋 Facts (key = value, one per line)",
                    lines=15,
                    placeholder="Hero = Arjun\nCity = Neo Mumbai\n...",
                )

        with gr.Row():
            gen_btn = gr.Button("🎲 Generate Document", variant="secondary")
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            run_btn = gr.Button("🚀 Run Comparison", variant="primary")

        gr.Markdown("---")
        gr.Markdown("## Results")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📦 RAG Story")
                rag_story_box = gr.Textbox(label="RAG Output", lines=12, interactive=False)
                rag_score_box = gr.Markdown(label="RAG Score")
            with gr.Column():
                gr.Markdown("### 🧠 RLM Story")
                rlm_story_box = gr.Textbox(label="RLM Output", lines=12, interactive=False)
                rlm_score_box = gr.Markdown(label="RLM Score")

        # Wire events
        gen_btn.click(
            fn=generate_document,
            inputs=[seed_input],
            outputs=[document_box, facts_box],
        )

        run_btn.click(
            fn=run_comparison,
            inputs=[document_box, facts_box],
            outputs=[rag_story_box, rlm_story_box, rag_score_box, rlm_score_box],
        )

    return app


if __name__ == "__main__":
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    app = build_ui()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
