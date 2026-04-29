"""Summary Agent — iteratively summarizes chunks of context."""

from __future__ import annotations

from typing import Callable

from layer5_context_access.chunker import fixed_windows
from layer1_input.context_repr import MountedContext


class SummaryAgent:
    def __init__(self, llm: Callable[[str], str], max_chunk_chars: int = 100_000):
        self.llm = llm
        self.max_chunk_chars = max_chunk_chars

    def run(self, context: str, query: str) -> str:
        ctx = MountedContext(context)
        chunks = fixed_windows(ctx, size=self.max_chunk_chars)
        
        if not chunks:
            return "No context provided."
            
        if len(chunks) == 1:
            prompt = f"Answer the query based on the context.\n\nContext:\n{chunks[0]}\n\nQuery: {query}"
            return self.llm(prompt)
            
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"You are iteratively reading a long document to answer: '{query}'. "
                f"Summarize the relevant information from this chunk (Chunk {i+1}/{len(chunks)}).\n\n"
                f"Chunk:\n{chunk}"
            )
            summaries.append(self.llm(prompt))
            
        final_context = "\n\n".join(f"Summary {i+1}:\n{s}" for i, s in enumerate(summaries))
        final_prompt = (
            f"Based on the following summaries of a larger document, answer the query.\n\n"
            f"Summaries:\n{final_context}\n\nQuery: {query}"
        )
        return self.llm(final_prompt)
