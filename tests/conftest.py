# Databricks notebook source
"""Collection guards for optional scientific-computing dependencies."""
from __future__ import annotations

import atexit
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# The jetflow-video package ships as a repo sub-tree (packages/jetflow-video),
# not as an installed dependency; make it importable for every test that
# reaches for it (same discipline as benchmarks/run_jetflow_distill_benchmark.py).
_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "packages" / "jetflow-video"
if _PACKAGE_DIR.is_dir() and str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))


def pytest_ignore_collect(collection_path, config) -> bool:
    name = getattr(collection_path, "name", None)
    if callable(name):
        name = collection_path.basename
    if name in {"test_neural_discovery.py", "test_jev_backend.py"}:
        return importlib.util.find_spec("jax") is None
    return False


# ---------------------------------------------------------------------------
# Tests do not write into the repository.
#
# The API's epistemic routes persist to ``APEDP_PERSISTENCE_DIR`` (default: the
# tracked ``apedp_persistence/``), so every run of the API tests rewrote a
# committed file. Each test process now works on a private copy.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if "APEDP_PERSISTENCE_DIR" not in os.environ:
    _persistence_copy = Path(tempfile.mkdtemp(prefix="apedp_persistence_"))
    _tracked_persistence = _REPO_ROOT / "apedp_persistence"
    if _tracked_persistence.is_dir():
        shutil.copytree(_tracked_persistence, _persistence_copy, dirs_exist_ok=True)
    os.environ["APEDP_PERSISTENCE_DIR"] = str(_persistence_copy)
    atexit.register(shutil.rmtree, _persistence_copy, ignore_errors=True)


# ---------------------------------------------------------------------------
# A missing native build is reported as a skip, never as a pass or a failure.
#
# Several engines exist only in the Rust kernel or need torch. On a machine
# without them a test can neither pass nor fail honestly: it did not run. Such
# a test is reported as skipped with the reason. Where the builds are supposed
# to be present -- CI sets APEDP_REQUIRE_NATIVE=1 -- nothing is converted, so a
# kernel that stopped building is a red run there, not a quiet column of skips.
# ---------------------------------------------------------------------------
_OPTIONAL_NATIVE_MODULES = frozenset({"apedp_kernel", "torch"})


def native_builds_required() -> bool:
    return os.environ.get("APEDP_REQUIRE_NATIVE", "").strip().lower() in {"1", "true", "yes"}


def missing_native_reason(exc: BaseException | None) -> str | None:
    """Why ``exc`` means "a native build is absent", or None if it means anything else.

    Follows the ``raise ... from`` / implicit-context chain, so an error that
    was wrapped on its way up is still recognised by its origin.
    """
    from apedp_dyn.kernel_shim import NativeKernelUnavailableError

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, NativeKernelUnavailableError):
            return f"native kernel not built: {exc}"
        if isinstance(exc, ModuleNotFoundError) and (exc.name or "").split(".")[0] in _OPTIONAL_NATIVE_MODULES:
            return f"optional native dependency not installed: {exc.name}"
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.excinfo is None or native_builds_required():
        return
    reason = missing_native_reason(call.excinfo.value)
    if reason is not None:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {reason}")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or native_builds_required():
        return
    # A collection error carries only its formatted traceback, so the missing
    # module is recognised from the interpreter's own message.
    text = str(report.longrepr)
    for module in sorted(_OPTIONAL_NATIVE_MODULES):
        if f"No module named '{module}'" in text:
            report.outcome = "skipped"
            report.longrepr = (
                str(collector.path), 0, f"Skipped: optional native dependency not installed: {module}"
            )
            return


# --------------------------------------------------------------------------
# Import-boundary evidence
# --------------------------------------------------------------------------
# tests/test_referee_core_boundary.py runs the referee core's tests in a child
# pytest and needs to know which modules that run loaded. Reading
# ``sys.modules`` at the end of the child's session is the direct measurement;
# the child asks for it by naming a file here. Unset (every ordinary run), this
# does nothing.
RECORD_LOADED_MODULES_ENV = "APEDP_RECORD_LOADED_MODULES"


def pytest_sessionfinish(session: pytest.Session) -> None:
    target = os.environ.get(RECORD_LOADED_MODULES_ENV)
    if not target or hasattr(session.config, "workerinput"):  # not from an xdist worker
        return
    Path(target).write_text("\n".join(sorted(sys.modules)) + "\n", encoding="utf-8")


# The video comparison module (packages/jetflow-video/jetflow_video/head_to_head.py)
# keeps process-wide state that two frozen test files share, and both leaks were
# found on stage on 27 November 2026 once the suite ran to its end again.
#
# 1. One call counter per model, in a dictionary keyed by ``id(model)`` that never
#    forgets one. A preregistered job builds one model in one process, so nothing
#    is wrong there. A test process builds many: when a model is freed, the next
#    one can be given the same id, finds the dead model's counter under it and
#    counts no calls. tests/test_video_hybrid_prereg.py failed that way (Python
#    3.11, run 45996673701: "calls_per_clip == 0.0"); it keeps only its own models
#    alive, not those of the tests before it.
# 2. The clip size. The larger test sets ``FRAMES``, ``SIZE`` and JetFlow's width
#    on the module (``run_prereg_video_hybrid.set_size``), for good. Every test of
#    tests/test_video_flow_head_to_head.py that then runs in the same process
#    fails ("size of tensor a (16) must match the size of tensor b (8)"). Which
#    tests share a process is decided by pytest-xdist at run time, so it passed on
#    some pushes and would have failed on others.
#
# The module and both test files are frozen by the runs' digests, so the cure is
# here: neither a counter nor a size outlives the test that made it.
VIDEO_COMPARISON = "jetflow_video.head_to_head"
VIDEO_SETTINGS = ("FRAMES", "SIZE", "JETFLOW")


def forget_video_call_counters() -> int:
    """Empty the video comparison module's call counters; how many there were (0 if it is not loaded)."""
    counters = getattr(sys.modules.get(VIDEO_COMPARISON), "_COUNTERS", None)
    if not counters:
        return 0
    forgotten = len(counters)
    counters.clear()
    return forgotten


@pytest.fixture(autouse=True)
def _video_comparison_state_does_not_outlive_a_test():
    forget_video_call_counters()
    module = sys.modules.get(VIDEO_COMPARISON)
    before = {name: getattr(module, name) for name in VIDEO_SETTINGS if hasattr(module, name)}
    yield
    if module is not None and sys.modules.get(VIDEO_COMPARISON) is module:
        for name, value in before.items():
            setattr(module, name, value)
