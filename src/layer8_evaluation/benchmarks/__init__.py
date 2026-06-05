from layer8_evaluation.benchmarks.browsecomp import BrowseCompTask, load_browsecomp_tasks
from layer8_evaluation.benchmarks.oolong import OolongTask, load_oolong_tasks
from layer8_evaluation.benchmarks.oolong_pairs import OolongPairsTask, load_oolong_pairs_tasks
from layer8_evaluation.benchmarks.s_niah import SNIAHExample, generate_sniah_tasks
from layer8_evaluation.benchmarks.codeqa import CodeQATask, load_codeqa_tasks


def trivial_example() -> SNIAHExample:
    """Return a minimal S-NIAH example for smoke-testing."""
    return SNIAHExample(
        haystack="Some filler text. The ANSWER is 42. More filler.",
        needle="What is the answer?",
        expected_span="42",
    )


def sample_task() -> OolongTask:
    """Return a simple benchmark task for unit tests."""
    return OolongTask(
        question="What is the answer?",
        gold="ANSWER",
        context="Context with the ANSWER embedded.",
    )


__all__ = [
    "BrowseCompTask",
    "load_browsecomp_tasks",
    "OolongTask",
    "load_oolong_tasks",
    "OolongPairsTask",
    "load_oolong_pairs_tasks",
    "SNIAHExample",
    "generate_sniah_tasks",
    "CodeQATask",
    "load_codeqa_tasks",
    "sample_task",
    "trivial_example",
]
