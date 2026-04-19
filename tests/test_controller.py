from layer1_input import MountedContext, normalize
from layer2_controller import RootController
from layer3_execution import RuntimeEngine, StateStore, ToolInterface
from layer6_output import OutputManager
from layer7_control import StepLimiter


def test_controller_round():
    steps = StepLimiter(max_steps=10)
    rt = RuntimeEngine(StateStore(), ToolInterface(), step_limiter=steps)
    ctrl = RootController(rt, output=OutputManager())
    ctx = MountedContext(text=normalize("ping"))
    r = ctrl.run_round(ctx, instruction="test")
    assert r["ok"] is True
