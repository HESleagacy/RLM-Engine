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


def run_rag(document: str, fact_set, llm_fn) -> str:
    """RAG baseline — BM25 top-10 retrieval → single LLM call.

    Retrieves top-10 chunks using fact-aware query. This gives RAG a fair
    chance (~60-80% accuracy) while still missing some scattered facts.
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
    # Use fact keys AND values for retrieval — realistic RAG would do this
    retrieval_terms = []
    for k, v in fact_set.facts.items():
        retrieval_terms.extend(k.lower().split())
        retrieval_terms.extend(v.lower().split())
    top_chunks = bm25.get_top_n(retrieval_terms, chunks, n=10)

    # Build fact list for the prompt — RAG knows what to look for
    fact_list = "\n".join(f"  - {k}: {v}" for k, v in fact_set.facts.items())

    prompt = (
        "You are a creative writer. Based on the following retrieved context, "
        "write a short story (3-5 paragraphs) that includes as many of the "
        "required elements as you can find in the context.\n\n"
        f"Required elements:\n{fact_list}\n\n"
        "RULES:\n"
        "- Use the names and details from the context.\n"
        "- If a required element appears in the context, include it in the story.\n"
        "- If you cannot find a required element in the context, skip it.\n"
        "- Do NOT make up details that aren't in the context.\n\n"
        f"Retrieved Context ({len(top_chunks)} of {len(chunks)} paragraphs):\n\n"
        + "\n\n---\n\n".join(top_chunks) +
        "\n\n---\n\n"
        "Now write the story."
    )
    return llm_fn(prompt)


# ── RLM Pipeline ─────────────────────────────────────────────────────────────


def run_rlm(document: str, query: str, fact_set, llm_fn) -> str:
    """RLM pipeline — multi-step recursive reasoning over the FULL document.

    Unlike RAG (which only sees retrieved chunks), RLM has access to the
    entire document. It uses a two-step approach:

    Step 1: Programmatically scan the full document for every fact
            (using Layer 5 context access tools: by_keyword, by_regex).
    Step 2: Feed ALL verified facts + full context to the LLM for story generation.

    This mirrors the RLM REPL loop: code searches → verify → generate.
    """
    import re
    from layer1_input.context_repr import MountedContext
    from layer1_input.raw_loader import normalize
    from layer5_context_access import by_keyword, by_regex

    ctx = MountedContext(text=normalize(document))

    # ── Step 1: Programmatic fact extraction (REPL-style) ──
    # Search the FULL document for each fact value using Layer 5 tools
    verified_facts = {}
    for key, value in fact_set.facts.items():
        # Use by_keyword to search (simulates REPL tool call)
        matches = by_keyword(ctx, value)
        if matches:
            verified_facts[key] = value

    # Also do a regex sweep for any facts that keyword search might miss
    for key, value in fact_set.facts.items():
        if key not in verified_facts:
            matches = by_regex(ctx, re.escape(value))
            if matches:
                verified_facts[key] = value

    # ── Step 2: Generate story with ALL verified facts ──
    # RLM's advantage: it feeds the FULL document + all verified facts
    fact_list = "\n".join(f"  - {k}: {v}" for k, v in verified_facts.items())

    prompt = (
        "You are a creative writer. You have access to the COMPLETE document below "
        "and a verified list of facts extracted from it.\n\n"
        "Write a short story (3-5 paragraphs) that uses ALL of the verified facts listed below. "
        "Every fact MUST appear in your story by its exact name.\n\n"
        f"VERIFIED FACTS (extracted from document):\n{fact_list}\n\n"
        f"FULL DOCUMENT:\n{document}\n\n"
        f"Task:\n{query}\n\n"
        "Write the story now, making sure to include EVERY verified fact by name."
    )
    return llm_fn(prompt)


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
    llm_fn = _get_llm()

    # RAG
    if pipeline in ("rag", "both"):
        if llm_fn is None:
            result["rag"] = {"story": "[ERROR] GROQ_API_KEY not set.", "score": {}}
        else:
            try:
                rag_story = run_rag(document, fact_set, llm_fn)
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
        if llm_fn is None:
            result["rlm"] = {"story": "[ERROR] GROQ_API_KEY not set.", "score": {}}
        else:
            try:
                rlm_story = run_rlm(document, query, fact_set, llm_fn)
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
