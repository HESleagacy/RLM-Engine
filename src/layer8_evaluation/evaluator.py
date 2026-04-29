"""Robust evaluation runner over benchmark tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Callable

from layer8_evaluation.metrics.accuracy import exact_match
from layer8_evaluation.metrics.cost import total_cost


@dataclass(frozen=True, slots=True)
class EvalResult:
    task_id: int
    correct: bool
    predicted: str
    gold: str
    cost_tokens: int
    cost_steps: int


def run_benchmark(
    dataset_name: str,
    tasks: list[Any],  # e.g. list[SNIAHExample], etc.
    run_fn: Callable[[str, str], tuple[str, int, int]], # context, query -> (answer, tokens, steps)
    output_path: str,
) -> None:
    """
    Run the provided model/agent function `run_fn` over the list of `tasks`.
    Logs the output to JSONL.
    """
    results = []
    
    with open(output_path, "w") as f:
        for i, task in enumerate(tasks):
            print(f"[{dataset_name}] Running task {i+1}/{len(tasks)}...")
            
            # Determine fields based on the generic dataclasses
            ctx = getattr(task, "haystack", getattr(task, "context", ""))
            query = getattr(task, "needle", getattr(task, "question", ""))
            gold = getattr(task, "expected_span", getattr(task, "gold", ""))
            
            # Run the agent
            try:
                pred, tokens, steps = run_fn(ctx, query)
            except Exception as e:
                print(f"[{dataset_name}] Task {i+1} failed: {e}")
                pred, tokens, steps = f"ERROR: {e}", 0, 0
                
            # Compare
            ok = exact_match(pred, gold) if gold else False
            
            res = EvalResult(
                task_id=i,
                correct=ok,
                predicted=pred,
                gold=str(gold),
                cost_tokens=tokens,
                cost_steps=steps,
            )
            
            results.append(res)
            
            # Log to file
            f.write(json.dumps(asdict(res)) + "\n")
            f.flush()
            
    accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0
    avg_tokens = sum(r.cost_tokens for r in results) / len(results) if results else 0
    avg_steps = sum(r.cost_steps for r in results) / len(results) if results else 0
    
    print(f"\n--- {dataset_name} Results ---")
    print(f"Accuracy:   {accuracy:.2%}")
    print(f"Avg Tokens: {avg_tokens:,.0f}")
    print(f"Avg Steps:  {avg_steps:.1f}")
    print("--------------------------\n")
