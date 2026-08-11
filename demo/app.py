"""Consistency Challenge Demo for the RLM pipeline.

Launch:
    pip install -e ".[demo]"
    PYTHONPATH=src GROQ_API_KEY=... python demo/app.py

Opens at http://localhost:7860
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

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


def run_rlm(document: str, query: str) -> str:
    """Run the production RootController REPL over the complete document."""
    from main import build_system, default_config_path
    from layer1_input.context_repr import MountedContext
    from layer1_input.raw_loader import normalize

    controller, _ = build_system(default_config_path(), use_groq=True)
    context = MountedContext(text=normalize(document))
    return controller.run_until_done(context, query).text


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

    fact_set, document = build_scattered_document(seed=seed)
    return jsonify({"document": document, "facts": fact_set.facts})


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    document = data.get("document", "")
    facts = data.get("facts", {})
    if not isinstance(document, str) or not document.strip() or len(document) > MAX_DOCUMENT_CHARS:
        return jsonify({"error": f"document must be at most {MAX_DOCUMENT_CHARS} characters"}), 400
    if not isinstance(facts, dict) or not facts or len(facts) > MAX_FACTS:
        return jsonify({"error": "missing document or facts"}), 400
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or len(key) > 200
        or len(value) > 500
        for key, value in facts.items()
    ):
        return jsonify({"error": "facts must be a mapping of short strings"}), 400
    if not PIPELINE_SLOTS.acquire(blocking=False):
        return jsonify({"error": "too many concurrent requests"}), 503

    try:
        fact_set = FactSet(facts=facts)
        try:
            story = run_rlm(document, build_query(fact_set))
            score = ConsistencyResult.from_story(story, fact_set)
            return jsonify(
                {
                    "story": story,
                    "score": {
                        "accuracy": score.accuracy,
                        "present": score.present,
                        "missing": score.missing,
                    },
                }
            )
        except Exception:
            return jsonify({"error": "RLM pipeline failed"}), 502
    finally:
        PIPELINE_SLOTS.release()


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    print("RLM Consistency Demo: http://localhost:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
