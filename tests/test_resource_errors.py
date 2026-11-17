"""Fault injection only: never fill a real disk or exhaust the test host's RAM."""

import errno
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading

import pytest

from document_ingestion import convert_pdf
from document_ingestion.text_extractor import ExtractedPage
from layer1_input import MountedContext
from layer2_controller.code_generator import CodeGenerator
from layer3_execution import RuntimeEngine, StateStore, ToolInterface
from shared.resource_errors import ResourceExhaustedError, WorkerTerminatedError, resource_error
from shared.types import ExecutionResult

pdf_module = importlib.import_module("document_ingestion.pdf_to_text")
main_module = importlib.import_module("main")
SOURCE = Path(__file__).parent / "fixtures" / "pdf1" / "Mathematics_of_LLMs_2026.pdf"
pytestmark = pytest.mark.resources


@pytest.mark.parametrize(("failure", "code"), [
    (MemoryError(), "out_of_memory"),
    (OSError(errno.ENOMEM, "test allocation failure"), "out_of_memory"),
    (OSError(errno.ENOSPC, "test full disk"), "disk_full"),
    (OSError(errno.EDQUOT, "test quota"), "disk_full"),
])
def test_wrapped_resource_causes_have_stable_actionable_errors(failure, code):
    wrapper = ValueError("Generic parser error must not hide the resource failure")
    wrapper.__cause__ = failure
    translated = resource_error(wrapper)
    assert translated.code == code
    assert "memory" in str(translated) if code == "out_of_memory" else "storage" in str(translated)
    assert "test " not in str(translated)


@pytest.mark.parametrize("failure", [
    OSError(errno.EACCES, "denied"), FileNotFoundError("missing"), ValueError("bad PDF"),
])
def test_unrelated_errors_are_not_misclassified_as_resource_exhaustion(failure):
    assert resource_error(failure) is None


@pytest.mark.parametrize("phase", ["cache-directory", "content-write", "content-close", "manifest", "publish"])
def test_disk_full_during_conversion_leaves_no_completed_or_temporary_artifact(tmp_path, monkeypatch, phase):
    cache = tmp_path / "cache"
    original_open = Path.open
    original_mkdir = Path.mkdir
    original_write_text = Path.write_text
    original_rename = Path.rename

    def full():
        raise OSError(errno.ENOSPC, "simulated disk full")

    class FullWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, text):
            if phase == "content-write":
                full()
            return self.stream.write(text)

        def __exit__(self, *args):
            self.stream.close()
            if phase == "content-close":
                full()

    def open_file(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path.name == "content.txt" and mode == "w" and phase.startswith("content-"):
            return FullWriter(stream)
        return stream

    def mkdir(path, *args, **kwargs):
        if phase == "cache-directory" and path == cache:
            full()
        return original_mkdir(path, *args, **kwargs)

    def write_text(path, *args, **kwargs):
        if phase == "manifest" and path.name == "manifest.json":
            full()
        return original_write_text(path, *args, **kwargs)

    def rename(path, *args, **kwargs):
        if phase == "publish" and path.name.startswith(".converting-"):
            full()
        return original_rename(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(ResourceExhaustedError) as caught:
        convert_pdf(SOURCE, cache, ocr_enabled=False)
    assert caught.value.code == "disk_full"
    assert "Free space" in str(caught.value)
    assert not cache.exists() or not list(cache.iterdir())


def test_memory_failure_after_one_page_cleans_up_partial_conversion(tmp_path, monkeypatch):
    def extract(*args, **kwargs):
        yield ExtractedPage(1, "Page one extracted successfully", True)
        raise MemoryError("simulated extraction allocation failure")

    monkeypatch.setattr(pdf_module, "extract_pages", extract)
    with pytest.raises(ResourceExhaustedError) as caught:
        convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    assert caught.value.code == "out_of_memory"
    assert not list((tmp_path / "cache").iterdir())


def test_parser_wrapping_memory_error_does_not_hide_the_cause(tmp_path, monkeypatch):
    import pypdf

    class Page:
        def extract_text(self):
            raise MemoryError()

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, source):
            pass

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    with pytest.raises(ResourceExhaustedError) as caught:
        convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    assert caught.value.code == "out_of_memory"
    assert "Could not extract" not in str(caught.value)


def test_cache_read_memory_failure_is_not_reported_as_corruption(tmp_path, monkeypatch):
    original = convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == original.manifest_path:
            raise OSError(errno.ENOMEM, "simulated allocation failure")
        return original_read(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", read)
        with pytest.raises(ResourceExhaustedError) as caught:
            convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False)
    assert caught.value.code == "out_of_memory"
    assert "Invalid cached artifact" not in str(caught.value)
    assert convert_pdf(SOURCE, tmp_path / "cache", ocr_enabled=False).cached


@pytest.mark.parametrize("threshold", [0, 1_048_576], ids=["disk-index", "ram-read"])
def test_mount_allocation_failure_is_reported_in_both_backends(tmp_path, monkeypatch, threshold):
    source = tmp_path / "source.txt"
    source.write_text("context", encoding="utf-8")
    original_open = Path.open

    def open_file(path, *args, **kwargs):
        if path == source:
            raise MemoryError()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(ResourceExhaustedError) as caught:
        MountedContext.from_file(source, memory_threshold_bytes=threshold)
    assert caught.value.code == "out_of_memory"


@pytest.mark.parametrize("timeout", [None, 5], ids=["inline", "worker"])
@pytest.mark.parametrize(("failure", "code"), [
    (MemoryError(), "out_of_memory"),
    (OSError(errno.ENOSPC, "full"), "disk_full"),
])
def test_execution_returns_resource_code_and_does_not_persist_partial_state(timeout, failure, code):
    state, tools = StateStore(), ToolInterface()
    state.set("original", "preserved")

    def fail():
        raise failure

    tools.register("fail", fail)
    runtime = RuntimeEngine(state, tools, timeout_seconds=timeout)
    result = runtime.execute("partial = 'not committed'\nresult = fail()")
    assert not result.ok
    assert result.error_code == code
    assert result.error
    assert result.value is None
    assert state.as_dict() == {"original": "preserved"}


def test_worker_serialization_memory_failure_returns_small_error_payload():
    class CannotSerialize:
        def __reduce__(self):
            raise MemoryError()

    state = StateStore()
    state.set("object", CannotSerialize())
    runtime = RuntimeEngine(state, ToolInterface(), timeout_seconds=5)
    result = runtime.execute("result = object")
    assert not result.ok
    assert result.error_code == "out_of_memory"


@pytest.mark.parametrize("cause", ["exit", "signal"])
def test_worker_death_is_reported_without_falsely_claiming_oom(cause):
    tools = ToolInterface()

    def terminate_this_worker():
        if cause == "signal":
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(17)

    tools.register("terminate", terminate_this_worker)
    result = RuntimeEngine(StateStore(), tools, timeout_seconds=5).execute("terminate()")
    assert not result.ok
    assert result.error_code == "worker_terminated"
    assert "without returning a result" in result.error
    assert "cannot be confirmed" in result.error
    assert ("-9" if cause == "signal" else "17") in result.error


def test_fork_memory_failure_returns_error_and_closes_unstarted_process(monkeypatch):
    import multiprocessing

    closed = []

    class CannotStart:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise OSError(errno.ENOMEM, "simulated fork allocation failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(multiprocessing.get_context("fork"), "Process", CannotStart)
    result = RuntimeEngine(StateStore(), ToolInterface()).execute("result = 1")
    assert result.error_code == "out_of_memory"
    assert closed == [True]


def test_parent_receiving_payload_memory_failure_returns_resource_error(monkeypatch):
    from multiprocessing.connection import Connection

    def receive(connection):
        raise MemoryError()

    monkeypatch.setattr(Connection, "recv", receive)
    result = RuntimeEngine(StateStore(), ToolInterface(), timeout_seconds=5).execute("result = 42")
    assert not result.ok
    assert result.error_code == "out_of_memory"


def test_validation_memory_failure_is_classified(monkeypatch):
    module = importlib.import_module("layer3_execution.runtime_engine")

    def validate(code):
        raise MemoryError()

    monkeypatch.setattr(module, "validate_code", validate)
    result = RuntimeEngine(StateStore(), ToolInterface()).execute("result = 1")
    assert result.error_code == "out_of_memory"


def test_controller_stops_after_resource_failure_instead_of_making_more_llm_calls():
    controller, _ = main_module.build_system(main_module.default_config_path())
    calls = []

    def fail():
        raise MemoryError()

    def chat(*args, **kwargs):
        calls.append(True)
        return json.dumps({"action": "execute", "code": "result = fail()"}), 0

    controller.runtime.tools.register("fail", fail)
    controller.codegen = CodeGenerator(chat=chat)
    outcome = controller.run_until_done(MountedContext("context"), "query")
    assert outcome.status == "failed"
    assert outcome.error_code == "out_of_memory"
    assert outcome.answer is None
    assert calls == [True]
    assert controller.output.intermediate.all() == ()


def test_controller_passes_worker_death_to_caller(monkeypatch):
    controller, _ = main_module.build_system(main_module.default_config_path())
    responses = iter([json.dumps({"action": "execute", "code": "result = 1"})])
    controller.codegen = CodeGenerator(chat=lambda *args, **kwargs: (next(responses), 0))
    monkeypatch.setattr(controller.runtime, "execute", lambda code: ExecutionResult(
        ok=False, error="Worker exited without a result; cause unknown",
        error_code="worker_terminated",
    ))
    outcome = controller.run_until_done(MountedContext("context"), "query")
    assert outcome.status == "failed"
    assert outcome.error_code == "worker_terminated"
    assert outcome.answer is None


def test_codegen_memory_failure_is_not_returned_as_a_successful_final_answer():
    controller, _ = main_module.build_system(main_module.default_config_path())

    def chat(*args, **kwargs):
        raise MemoryError()

    controller.codegen = CodeGenerator(chat=chat)
    outcome = controller.run_until_done(MountedContext("context"), "query")
    assert outcome.status == "failed"
    assert outcome.error_code == "out_of_memory"
    assert outcome.answer is None


@pytest.mark.parametrize("target", ["pdf", "engine"])
@pytest.mark.parametrize(("failure", "code"), [
    (MemoryError(), "out_of_memory"), (OSError(errno.ENOSPC, "full"), "disk_full"),
])
def test_clis_exit_nonzero_with_actionable_errors_not_tracebacks(monkeypatch, capsys, target, failure, code):
    def fail(*args, **kwargs):
        raise failure

    if target == "pdf":
        monkeypatch.setattr(pdf_module, "file_digest", fail)
        monkeypatch.setattr(sys, "argv", ["rlm-pdf", str(SOURCE), "--no-ocr"])
        command = pdf_module.main
    else:
        monkeypatch.setattr(MountedContext, "from_file", fail)
        monkeypatch.setattr(sys, "argv", ["rlm", "--context-file", str(SOURCE)])
        command = main_module.main
    with pytest.raises(SystemExit) as caught:
        command()
    captured = capsys.readouterr()
    assert caught.value.code == 1
    assert f"[{code}]" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("code", ["out_of_memory", "worker_terminated"])
def test_engine_cli_does_not_print_resource_failure_as_success(monkeypatch, capsys, code):
    class Controller:
        def run_round(self, *args, **kwargs):
            return {"ok": False, "execution": ExecutionResult(
                ok=False, error="Execution failed.", error_code=code
            )}

    monkeypatch.setattr(main_module, "build_system", lambda *args, **kwargs: (Controller(), None))
    monkeypatch.setattr(sys, "argv", ["rlm", "--prompt", "test"])
    with pytest.raises(SystemExit) as caught:
        main_module.main()
    output = capsys.readouterr()
    assert caught.value.code == 1
    assert f"[{code}]" in output.err
    assert not output.out


def test_pdf_cli_reports_memory_failure_before_conversion_starts(monkeypatch, capsys):
    def configure(*args, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(pdf_module, "TesseractOCR", configure)
    monkeypatch.setattr(sys, "argv", ["rlm-pdf", str(SOURCE)])
    with pytest.raises(SystemExit) as caught:
        pdf_module.main()
    assert caught.value.code == 1
    assert "[out_of_memory]" in capsys.readouterr().err


@pytest.mark.parametrize(("failure", "code", "status"), [
    (MemoryError(), "out_of_memory", 503),
    (OSError(errno.ENOSPC, "full"), "disk_full", 503),
    (WorkerTerminatedError("Worker terminated; cause unknown"), "worker_terminated", 502),
])
def test_web_api_returns_structured_failure_and_releases_request_slot(monkeypatch, failure, code, status):
    demo = importlib.import_module("demo.app")
    monkeypatch.setattr(demo, "PIPELINE_SLOTS", threading.BoundedSemaphore(1))

    def fail(*args):
        raise failure

    monkeypatch.setattr(demo, "run_rlm", fail)
    with demo.app.test_client() as client:
        payload = {"document": "Hero Arjun", "facts": {"Hero": "Arjun"}}
        response = client.post("/api/run", json=payload)
        assert response.status_code == status
        assert response.get_json()["code"] == code
        assert "story" not in response.get_json()
        monkeypatch.setattr(demo, "run_rlm", lambda *args: "Hero Arjun.")
        assert client.post("/api/run", json=payload).status_code == 200


def test_document_generation_api_reports_memory_failure(monkeypatch):
    demo = importlib.import_module("demo.app")

    def fail(*args, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(demo, "build_scattered_document", fail)
    with demo.app.test_client() as client:
        response = client.post("/api/generate", json={"seed": 42})
    assert response.status_code == 503
    assert response.get_json()["code"] == "out_of_memory"


def test_run_api_reports_memory_failure_while_parsing_request(monkeypatch):
    demo = importlib.import_module("demo.app")

    def parse(*args, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(demo.app.request_class, "get_json", parse)
    with demo.app.test_client() as client:
        response = client.post("/api/run", json={})
    assert response.status_code == 503
    assert response.get_json()["code"] == "out_of_memory"
