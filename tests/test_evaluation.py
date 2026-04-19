from layer8_evaluation import evaluate_one
from layer8_evaluation.benchmarks import sample_task, trivial_example
from layer8_evaluation.metrics import exact_match


def test_metrics():
    assert exact_match("a", "a")
    ex = trivial_example()
    assert "ANSWER" in ex.haystack
    t = sample_task()
    r = evaluate_one(t.gold, lambda: t.gold, tokens=1, steps=1)
    assert r.correct
