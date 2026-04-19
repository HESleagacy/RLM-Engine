from layer1_input import MountedContext, byte_length, describe, normalize


def test_normalize_and_mount():
    t = normalize("  a\r\nb  ")
    ctx = MountedContext(text=t)
    assert ctx.text == "a\nb"
    assert byte_length(ctx) > 0
    meta = describe(ctx)
    assert meta.line_count == 2
