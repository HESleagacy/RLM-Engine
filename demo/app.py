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
import threading
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
MAX_DOCUMENT_CHARS = 250_000
MAX_FACTS = 50
PIPELINE_SLOTS = threading.BoundedSemaphore(2)


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


def run_rag(document: str, query: str, fact_set, llm_fn) -> str:
    """RAG baseline — BM25 top-10 retrieval → single LLM call.

    Retrieves top-10 chunks using the same task facts available to the caller.
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
    # Retrieval has the same fact vocabulary as the task query.
    retrieval_terms = []
    for k, v in fact_set.facts.items():
        retrieval_terms.extend(k.lower().split())
        retrieval_terms.extend(v.lower().split())
    top_chunks = bm25.get_top_n(retrieval_terms, chunks, n=10)

    prompt = (
        "You are a creative writer. Based only on the retrieved context, answer "
        "the following task. Do not invent details that are not in the context.\n\n"
        "RULES:\n"
        "- Use the names and details from the context.\n"
        "- If a required element appears in the context, include it in the story.\n"
        "- If you cannot find a required element in the context, skip it.\n"
        "- Do NOT make up details that aren't in the context.\n\n"
        f"Retrieved Context ({len(top_chunks)} of {len(chunks)} paragraphs):\n\n"
        + "\n\n---\n\n".join(top_chunks) +
        "\n\n---\n\n"
        f"Task:\n{query}\n\nNow write the story."
    )
    return llm_fn(prompt)


# ── RLM Pipeline ─────────────────────────────────────────────────────────────


def run_rlm(document: str, query: str, fact_set, llm_fn) -> str:
    """Run the production RootController REPL over the complete document."""
    from main import build_system, default_config_path
    from layer1_input.context_repr import MountedContext
    from layer1_input.raw_loader import normalize
    controller, _ = build_system(default_config_path(), use_groq=True)
    return controller.run_until_done(MountedContext(text=normalize(document)), query).text


# ── API Routes ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"error": "seed must be an integer"}), 400
    fact_set, doc = build_scattered_document(seed=seed)
    return jsonify({
        "document": doc,
        "facts": fact_set.facts,
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    document = data.get("document", "")
    facts = data.get("facts", {})
    pipeline = data.get("pipeline", "both")  # "rag", "rlm", or "both"

    if not isinstance(document, str) or not document.strip() or len(document) > MAX_DOCUMENT_CHARS:
        return jsonify({"error": f"document must be at most {MAX_DOCUMENT_CHARS} characters"}), 400
    if not isinstance(facts, dict) or not facts or len(facts) > MAX_FACTS:
        return jsonify({"error": "Missing document or facts"}), 400
    if pipeline not in {"rag", "rlm", "both"}:
        return jsonify({"error": "pipeline must be rag, rlm, or both"}), 400
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or len(key) > 200
        or len(value) > 500
        for key, value in facts.items()
    ):
        return jsonify({"error": "facts must be a mapping of short strings"}), 400

    if not PIPELINE_SLOTS.acquire(blocking=False):
        return jsonify({"error": "too many concurrent pipeline requests"}), 503

    try:
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
                    rag_story = run_rag(document, query, fact_set, llm_fn)
                    rag_res = ConsistencyResult.from_story(rag_story, fact_set)
                    result["rag"] = {
                        "story": rag_story,
                        "score": {
                            "accuracy": rag_res.accuracy,
                            "present": rag_res.present,
                            "missing": rag_res.missing,
                        },
                    }
                except Exception:
                    result["rag"] = {"story": "[ERROR] pipeline failed", "score": {}}

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
                except Exception:
                    result["rlm"] = {"story": "[ERROR] pipeline failed", "score": {}}

        return jsonify(result)
    finally:
        PIPELINE_SLOTS.release()


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    print("🧠 RLM vs RAG Demo — http://localhost:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
