from layer7_control import BudgetManager, RecursionGuard, StepLimiter


def test_limits():
    s = StepLimiter(max_steps=2)
    s.tick()
    s.tick()
    assert not s.allow()

    b = BudgetManager(limit=10)
    b.spend(5)
    assert b.remaining() == 5

    g = RecursionGuard(max_depth=1)
    g.enter()
    try:
        g.enter()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected depth error")
