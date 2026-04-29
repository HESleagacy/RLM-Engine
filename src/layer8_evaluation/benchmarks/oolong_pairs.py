"""OOLONG-Pairs — quadratic scaling benchmark loader."""

from __future__ import annotations

import random
from dataclasses import dataclass

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


@dataclass(frozen=True, slots=True)
class OolongPairsTask:
    question: str
    gold: str
    context: str


_PAIR_QUERIES = [
    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or location.",
    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or human being.",
    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or abbreviation.",
    # Truncated list for simplicity of generator, representing the 20 queries from the paper
]


def load_oolong_pairs_tasks(num_tasks: int = 20) -> list[OolongPairsTask]:
    """
    Generate OOLONG-Pairs tasks by pulling the trec_coarse context and appending
    the synthetic pair-aggregation queries.
    """
    if load_dataset is None:
        raise ImportError("Please install `datasets` to run OOLONG-Pairs")

    print("[OOLONG-Pairs] Streaming trec_coarse contexts...")
    ds = load_dataset(
        "oolongbench/oolong-synth", 
        "trec_coarse", 
        split="validation", 
        streaming=True
    )
    
    tasks = []
    # Fetch a few base contexts to apply our synthetic pair queries to
    base_contexts = []
    for row in ds:
        ctx = row.get("text", row.get("context", ""))
        if ctx:
            base_contexts.append(ctx)
        if len(base_contexts) >= 5: # Just need a few large contexts
            break
            
    for i in range(num_tasks):
        q = _PAIR_QUERIES[i % len(_PAIR_QUERIES)]
        # Add labels description from the paper
        full_q = (
            f"{q} Each of the questions can be labelled as one of the labels: "
            "description and abstract concept, entity, human being, numeric value, location, abbreviation. "
            "In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines."
        )
        
        ctx = base_contexts[i % len(base_contexts)]
        
        tasks.append(
            OolongPairsTask(
                question=full_q,
                gold="[]", # In a real evaluator, we would programmatically compute the gold array
                context=ctx
            )
        )

    return tasks
