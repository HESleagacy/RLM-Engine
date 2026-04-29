"""OOLONG — linear scaling benchmark loader for OOLONG trec_coarse tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


@dataclass(frozen=True, slots=True)
class OolongTask:
    question: str
    gold: str
    context: str


def load_oolong_tasks(num_tasks: int = 50) -> list[OolongTask]:
    """
    Load the OOLONG trec_coarse validation tasks.
    Streams to avoid downloading the entire benchmark suite if large.
    """
    if load_dataset is None:
        raise ImportError("Please install `datasets` to run OOLONG")

    print("[OOLONG] Streaming trec_coarse tasks from oolongbench/oolong-synth...")
    
    # Use streaming to avoid pulling all subsets
    ds = load_dataset(
        "oolongbench/oolong-synth", 
        "trec_coarse", 
        split="validation", 
        streaming=True
    )
    
    tasks = []
    for i, row in enumerate(ds):
        if i >= num_tasks:
            break
            
        # OOLONG has a specific text context and question layout
        context_str = row.get("text", "")
        if not context_str and "context" in row:
            context_str = row["context"]
            
        q = row.get("question", "")
        a = row.get("answer", row.get("label", ""))
        
        tasks.append(
            OolongTask(
                question=q,
                gold=str(a),
                context=context_str
            )
        )

    return tasks
