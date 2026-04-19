from layer2_controller.code_generator import CodeGenerator


def test_llm_plain_text_becomes_result_assignment():
    def llm(_prompt: str) -> str:
        return "Just OK"

    cg = CodeGenerator(llm)
    code = cg.generate("do something", "ignored")
    ns: dict = {}
    exec(compile(code, "<t>", "exec"), ns, ns)
    assert ns["result"] == "Just OK"


def test_llm_fenced_python_preserved():
    def llm(_prompt: str) -> str:
        return "```python\nresult = 40 + 2\n```"

    cg = CodeGenerator(llm)
    code = cg.generate("x", "y")
    ns: dict = {}
    exec(compile(code, "<t>", "exec"), ns, ns)
    assert ns["result"] == 42
