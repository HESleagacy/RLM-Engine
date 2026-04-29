"""CodeAct + BM25 Agent — Python execution with BM25 retrieval."""

from __future__ import annotations

import re
from typing import Callable

from layer3_execution.runtime_engine import RuntimeEngine
from layer3_execution.state_store import StateStore
from layer3_execution.tool_interface import ToolInterface

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class CodeActBM25Agent:
    def __init__(self, llm: Callable[[str], str], max_steps: int = 10):
        self.llm = llm
        self.max_steps = max_steps
        
        self.state = StateStore()
        self.tools = ToolInterface()
        self.runtime = RuntimeEngine(self.state, self.tools, strict_sandbox=True)
        
    def _build_bm25(self, context: str) -> list[str]:
        # Simple line-by-line or paragraph split for BM25
        return [p.strip() for p in context.split("\n\n") if p.strip()]

    def run(self, context: str, query: str) -> str:
        if BM25Okapi is None:
            raise ImportError("Please install rank_bm25 to use CodeActBM25Agent")
            
        corpus = self._build_bm25(context)
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        def search(query_str: str, n: int = 5) -> str:
            tokenized_query = query_str.lower().split()
            top_docs = bm25.get_top_n(tokenized_query, corpus, n=n)
            return "\n---\n".join(top_docs)
            
        self.tools.register("SEARCH", search)
        
        system_prompt = (
            "You are a CodeAct agent. You can execute Python code and search documents.\n"
            "Format:\n"
            "THINK: your thought\n"
            "ACT: write ```python code``` or SEARCH(\"query\")\n"
            "ANSWER: your final answer"
        )
        
        history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        
        for _ in range(self.max_steps):
            # Flatten history for LLM if it's a simple string-based LLM, or assume it handles lists
            # Since we only have a generic Callable, we format it as text:
            prompt = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history)
            
            response = self.llm(prompt)
            history.append({"role": "assistant", "content": response})
            
            if "ANSWER:" in response:
                return response.split("ANSWER:")[-1].strip()
                
            code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                res = self.runtime.execute(code)
                output = res.stdout if res.ok else res.error
                history.append({"role": "user", "content": f"Code Execution Result:\n{output}"})
                continue
                
            search_match = re.search(r"SEARCH\([\"'](.*?)[\"']\)", response)
            if search_match:
                sq = search_match.group(1)
                res = search(sq)
                history.append({"role": "user", "content": f"Search Results:\n{res}"})
                continue
                
        return "Failed to find answer within max steps."
