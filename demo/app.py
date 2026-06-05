"""Consistency Challenge Demo — Flask API + JS frontend.

Launch:
    pip install flask
    PYTHONPATH=src GROQ_API_KEY=... python demo/app.py

Opens at http://localhost:7860
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from flask import Flask, jsonify, request, send_from_directory

from layer8_evaluation.benchmarks.consistency import (
    ConsistencyResult,
    FactSet,
    build_query,
    build_scattered_document,
)

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))


# ── LLM helpers ──────────────────────────────────────────────────────────────


def _get_llm():
    try:
        from shared.groq_client import make_groq_llm
        llm = make_groq_llm()
        def call(prompt: str) -> str:
            text, _ = llm(prompt, max_tokens=2048)
            return text
        return call
    except Exception:
        return None


# ── RAG Pipeline ─────────────────────────────────────────────────────────────


def run_rag(document: str, query: str, llm_fn) -> str:
    """RAG baseline — BM25 top-3 retrieval → single LLM call.

    Intentionally limited: retrieves only 3 chunks (not the full document),
    and the prompt does NOT list the required fact values — the LLM must
    rely solely on what BM25 retrieved.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return "[ERROR] rank_bm25 not installed."

    chunks = [p.strip() for p in document.split("\n\n") if p.strip()]
    if not chunks:
        return "[ERROR] No content."

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    # Only retrieve top-3 chunks — this is the core RAG limitation.
    # With 15 facts scattered across ~35 paragraphs, 3 chunks will
    # typically contain only 2-4 facts.
    top_chunks = bm25.get_top_n(query.lower().split(), chunks, n=3)

    # CRITICAL: Do NOT pass the full fact list to the LLM.
    # RAG's weakness is that it can only use what was retrieved.
    # If we gave it the fact names, the LLM would use prior knowledge to fill gaps.
    prompt = (
        "You are a creative writer. Below are some excerpts retrieved from a larger document. "
        "Write a short story (3-5 paragraphs) that incorporates ALL the factual details "
        "(character names, locations, objects, organizations) found in these excerpts.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY names, places, and details that explicitly appear in the excerpts below.\n"
        "- Do NOT invent any character names, place names, or object names.\n"
        "- If the excerpts mention a name or detail, you MUST include it.\n"
        "- Do NOT add any proper nouns that are not in the excerpts.\n\n"
        f"--- RETRIEVED EXCERPTS ---\n\n" + "\n\n---\n\n".join(top_chunks) +
        "\n\n--- END OF EXCERPTS ---\n\n"
        "Now write the story using only the details from the excerpts above."
    )
    return llm_fn(prompt)


# ── RLM Pipeline ─────────────────────────────────────────────────────────────


def run_rlm(document: str, query: str) -> str:
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

    steps = StepLimiter(max_steps=50)
    budget = BudgetManager(limit=50_000)
    guard = RecursionGuard(max_depth=3)
    state = StateStore()
    tools = ToolInterface()
    runtime = RuntimeEngine(state, tools, step_limiter=steps, strict_sandbox=True)

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

    sub_llm = make_groq_llm(model="llama-3.1-8b-instant")
    rec_manager = RecursionManager(guard=guard, llm=sub_llm, budget=budget)
    tools.register("llm_query", rec_manager.run_subtask)

    root_llm = make_groq_llm(model="llama-3.3-70b-versatile")
    root_chat = make_groq_chat(model="llama-3.3-70b-versatile")
    codegen = CodeGenerator(llm=root_llm, chat=root_chat, budget=budget)

    controller = RootController(runtime, codegen=codegen, max_rounds=10, stdout_truncation=3000)
    ctx = MountedContext(text=normalize(document))
    answer = controller.run_until_done(ctx, query)
    return answer.text


# ── API Routes ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    seed = int(data.get("seed", 42))
    fact_set, doc = build_scattered_document(seed=seed)
    return jsonify({
        "document": doc,
        "facts": fact_set.facts,
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.json or {}
    document = data.get("document", "")
    facts = data.get("facts", {})
    pipeline = data.get("pipeline", "both")  # "rag", "rlm", or "both"

    if not document or not facts:
        return jsonify({"error": "Missing document or facts"}), 400

    fact_set = FactSet(facts=facts)
    query = build_query(fact_set)

    result = {}

    # RAG
    if pipeline in ("rag", "both"):
        llm_fn = _get_llm()
        if llm_fn is None:
            result["rag"] = {"story": "[ERROR] GROQ_API_KEY not set.", "score": {}}
        else:
            try:
                rag_story = run_rag(document, query, llm_fn)
                rag_res = ConsistencyResult.from_story(rag_story, fact_set)
                result["rag"] = {
                    "story": rag_story,
                    "score": {
                        "accuracy": rag_res.accuracy,
                        "present": rag_res.present,
                        "missing": rag_res.missing,
                    },
                }
            except Exception as e:
                result["rag"] = {"story": f"[ERROR] {e}", "score": {}}

    # RLM
    if pipeline in ("rlm", "both"):
        try:
            rlm_story = run_rlm(document, query)
            rlm_res = ConsistencyResult.from_story(rlm_story, fact_set)
            result["rlm"] = {
                "story": rlm_story,
                "score": {
                    "accuracy": rlm_res.accuracy,
                    "present": rlm_res.present,
                    "missing": rlm_res.missing,
                },
            }
        except Exception as e:
            result["rlm"] = {"story": f"[ERROR] {e}", "score": {}}

    return jsonify(result)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    print("🧠 RLM vs RAG Demo — http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
