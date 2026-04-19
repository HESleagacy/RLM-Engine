from layer6_output import OutputManager


def test_finalize():
    om = OutputManager()
    om.intermediate.append("a")
    om.intermediate.append("b")
    fa = om.finalize_joined()
    assert fa.locked
    assert "a" in fa.text and "b" in fa.text
