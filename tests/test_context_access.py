from layer1_input import MountedContext
from layer5_context_access import by_keyword, fixed_windows, peek_head


def test_access():
    ctx = MountedContext(text="alpha\nbeta gamma\n")
    assert "beta" in peek_head(ctx, 20)
    assert any("beta" in ln for ln in by_keyword(ctx, "beta"))
    chunks = fixed_windows(ctx, size=5, overlap=0)
    assert len(chunks) >= 1
