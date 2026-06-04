from layer2_controller.code_generator import CodeGenerator


def test_llm_plain_text_becomes_result_assignment():
    def llm(_prompt: str, *, max_tokens: int | None = None) -> tuple[str, int]:
        return ("Just OK", 5)

    cg = CodeGenerator(llm)
    code = cg.generate("do something", "ignored")
    ns: dict = {}
    exec(compile(code, "<t>", "exec"), ns, ns)
    assert ns["result"] == "Just OK"


def test_llm_fenced_python_preserved():
    def llm(_prompt: str, *, max_tokens: int | None = None) -> tuple[str, int]:
        return ("```python\nresult = 40 + 2\n```", 10)

    cg = CodeGenerator(llm)
    code = cg.generate("x", "y")
    ns: dict = {}
    exec(compile(code, "<t>", "exec"), ns, ns)
    assert ns["result"] == 42
