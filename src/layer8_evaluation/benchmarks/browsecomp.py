"""BrowseComp — multi-hop reasoning benchmark loader for BrowseComp-Plus (1K documents)."""

from __future__ import annotations

import json
from dataclasses import dataclass

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


@dataclass(frozen=True, slots=True)
class BrowseCompTask:
    question: str
    gold: str
    context: str  # The 1000 aggregated documents


def load_browsecomp_tasks(num_tasks: int = 150) -> list[BrowseCompTask]:
    """
    Load BrowseComp-Plus queries and their associated 1000 documents context.
    Streams the dataset to avoid a 6GB+ local download.
    """
    if load_dataset is None:
        raise ImportError("Please install `datasets` (pip install datasets) to run BrowseComp")

    print("[BrowseComp] Streaming queries from Tevatron/browsecomp-plus...")
    # Load the queries (usually test split)
    queries_ds = load_dataset(
        "Tevatron/browsecomp-plus", 
        "default", 
        split="test", 
        streaming=True
    )

    print("[BrowseComp] Streaming corpus from Tevatron/browsecomp-plus-corpus...")
    # Corpus contains the actual documents
    corpus_ds = load_dataset(
        "Tevatron/browsecomp-plus-corpus", 
        split="train", 
        streaming=True
    )
    
    # In a true 1000-document setting per task, we would map the query to its specific
    # evidence docs and 990 hard negatives.
    # To prevent a massive cross-join while streaming, this is a simplified loader
    # that yields tasks with a combined context of documents.
    
    tasks = []
    # Grab the first 1000 documents from the corpus as our haystack for these tasks
    # (In the exact paper, they use the specific hard negatives per query).
    doc_iterator = iter(corpus_ds)
    docs = []
    try:
        for _ in range(1000):
            docs.append(next(doc_iterator)["text"])
    except StopIteration:
        pass
        
    combined_context = "\n\n".join(f"Document {i}:\n{text}" for i, text in enumerate(docs))

    for i, row in enumerate(queries_ds):
        if i >= num_tasks:
            break
        
        # De-obfuscate or extract the question and gold answer based on Tevatron's format
        # Usually it's in a 'query' or 'question' field, and 'answers' field.
        q = row.get("query", row.get("question", str(row)))
        a = row.get("answers", row.get("gold", "Unknown"))
        if isinstance(a, list) and len(a) > 0:
            a = a[0]
            
        tasks.append(
            BrowseCompTask(
                question=q,
                gold=str(a),
                context=combined_context
            )
        )

    return tasks
