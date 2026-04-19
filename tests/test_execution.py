from layer3_execution import RuntimeEngine, StateStore, ToolInterface
from layer7_control import StepLimiter


def test_runtime_sets_result():
    steps = StepLimiter(max_steps=5)
    rt = RuntimeEngine(StateStore(), ToolInterface(), step_limiter=steps)
    res = rt.execute("result = 41 + 1")
    assert res.ok
    assert res.value == 42
