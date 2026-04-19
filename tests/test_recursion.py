from layer4_recursion import RecursionManager
from layer7_control import RecursionGuard


def test_recursion_manager():
    def fake_llm(prompt: str) -> str:
        return f"echo:{prompt}"

    mgr = RecursionManager(RecursionGuard(max_depth=3), fake_llm)
    assert "echo:hi" in mgr.run_subtask("hi")
