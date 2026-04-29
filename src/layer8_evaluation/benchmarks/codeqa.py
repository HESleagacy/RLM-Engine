"""LongBench-v2 CodeQA — code repository understanding tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


@dataclass(frozen=True, slots=True)
class CodeQATask:
    question: str
    choices: dict[str, str]
    gold: str
    context: str


def load_codeqa_tasks(num_tasks: int = 50) -> list[CodeQATask]:
    """
    Load LongBench-v2 Code repository understanding tasks.
    Streams to avoid downloading the entire LongBench-v2 suite.
    """
    if load_dataset is None:
        raise ImportError("Please install `datasets` to run LongBench-v2 CodeQA")

    print("[CodeQA] Streaming Code repository understanding tasks from THUDM/LongBench-v2...")
    ds = load_dataset(
        "THUDM/LongBench-v2", 
        split="train", 
        streaming=True
    )
    
    tasks = []
    
    for row in ds:
        # Filter for Code repository understanding
        if row.get("domain", "") != "Code repository understanding":
            continue
            
        q = row.get("question", "")
        # LongBench-v2 has choices A, B, C, D
        choices = {
            "A": row.get("choice_A", ""),
            "B": row.get("choice_B", ""),
            "C": row.get("choice_C", ""),
            "D": row.get("choice_D", "")
        }
        
        # Build the final multiple-choice prompt
        full_q = f"You are a helpful assistant that can answer questions about code repositories. You must answer the given question: {q}\n"
        for k, v in choices.items():
            full_q += f"{k}: {v}\n"
            
        a = row.get("answer", "")
        ctx = row.get("context", "")
        
        tasks.append(
            CodeQATask(
                question=full_q,
                choices=choices,
                gold=str(a),
                context=ctx
            )
        )
        
        if len(tasks) >= num_tasks:
            break

    return tasks
