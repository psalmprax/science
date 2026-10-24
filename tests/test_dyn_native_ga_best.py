"""Native GA best-selection pins: best means minimum MSE, never index 0.

Regression test for the complexity-ordered frontier bug: the Rust engine
returns the Pareto frontier sorted by complexity, so taking element 0
reported the simplest candidate (the mean constant) instead of the best.
Fast (native path, no scipy).
"""

import re

import numpy as np
import pytest

from apedp.evolutionary_search import (
    NativeSearchUnavailableError,
    SearchConfig,
    evolve,
)


def test_native_best_is_minimum_mse_recovery():
    x = np.linspace(-3, 3, 200)
    y = 3.0 * x ** 2 + 2.0
    res = evolve({"x": x}, y, config=SearchConfig(
        use_native_ga=True, islands=4, island_size=50,
        generations=20, seed=42))
    mses = [p["train_mse"] for p in res.pareto_front]
    assert mses, "empty Pareto frontier"
    best_mse = min(mses)
    assert best_mse < 1e-6, f"native GA failed to recover x^2 (best MSE {best_mse})"
    # best_expression must be the min-MSE entry, not frontier position 0.
    entry = next(p for p in res.pareto_front if p["train_mse"] == best_mse)
    assert res.best_expression == entry["expression"]
    assert "x" in res.best_expression


# ---------------------------------------------------------------------------
# strict_native_ga: the Python fallback must never be taken silently.
# The two paths differ by orders of magnitude in cost, so a downgrade has to be
# an exception the caller sees, not a DEBUG log line.
# ---------------------------------------------------------------------------

_TINY = {"islands": 1, "island_size": 4, "generations": 1, "seed": 0,
             "tune_restarts": 0, "tune_iters": 1}


def _tiny_data(n=10):
    x = np.linspace(-1.0, 1.0, n)
    return {"x": x}, 2.0 * x + 1.0


def test_strict_requires_use_native_ga():
    x, y = _tiny_data()
    with pytest.raises(ValueError, match="requires use_native_ga=True"):
        evolve(x, y, config=SearchConfig(strict_native_ga=True, **_TINY))


def test_strict_raises_when_kernel_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "apedp.evolutionary_search.is_rust_kernel_available", lambda: False)
    x, y = _tiny_data()
    with pytest.raises(NativeSearchUnavailableError, match="cannot serve the search"):
        evolve(x, y, config=SearchConfig(
            use_native_ga=True, strict_native_ga=True, **_TINY))


def test_strict_raises_when_the_native_call_fails(monkeypatch):
    class _ExplodingBridge:
        def __init__(self, *args, **kwargs):
            pass

        def evolve_native(self, *args, **kwargs):
            raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr("apedp.rust_bridge.RustKernelBridge", _ExplodingBridge)
    x, y = _tiny_data()
    with pytest.raises(NativeSearchUnavailableError, match="simulated kernel failure"):
        evolve(x, y, config=SearchConfig(
            use_native_ga=True, strict_native_ga=True, **_TINY))


def test_non_strict_still_falls_back_to_python(monkeypatch):
    """The default behaviour is unchanged: fallback, but now observable."""
    class _ExplodingBridge:
        def __init__(self, *args, **kwargs):
            pass

        def evolve_native(self, *args, **kwargs):
            raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr("apedp.rust_bridge.RustKernelBridge", _ExplodingBridge)
    x, y = _tiny_data()
    res = evolve(x, y, config=SearchConfig(use_native_ga=True, **_TINY))
    assert "native_rust" not in res.history[0], (
        "the Python fallback path must not be reported as a native run")
    assert res.pareto_front, "fallback produced no frontier"


def test_enforcement_refuses_fallback_from_evolve(monkeypatch):
    """The process-wide switch closes the hole a forgotten flag would leave."""
    from apedp.rust_bridge import enforce_native_search

    monkeypatch.setattr(
        "apedp.evolutionary_search.is_rust_kernel_available", lambda: False)
    x, y = _tiny_data()
    enforce_native_search(True)
    try:
        with pytest.raises(NativeSearchUnavailableError,
                           match="enforce_native_search"):
            evolve(x, y, config=SearchConfig(use_native_ga=True, **_TINY))
        # A harness that never asked for the kernel is refused too, rather than
        # silently timed on the Python loop.
        with pytest.raises(ValueError, match="requires use_native_ga=True"):
            evolve(x, y, config=SearchConfig(**_TINY))
    finally:
        enforce_native_search(False)


def test_enforcement_refuses_fallback_from_the_bridge(monkeypatch):
    import apedp_kernel
    from apedp.rust_bridge import RustKernelBridge, enforce_native_search

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr(apedp_kernel, "evolve_native", _boom, raising=False)
    data, target = _tiny_data()
    columns = [data["x"]]
    enforce_native_search(True)
    try:
        with pytest.raises(NativeSearchUnavailableError,
                           match="simulated kernel failure"):
            RustKernelBridge().evolve_native(
                columns, ["x"], target, population_size=4, generations=1)
        # An explicitly requested fallback stays legal: the rust-vs-python
        # sweeps measure both paths on purpose.
        pareto, _ = RustKernelBridge(force_fallback=True).evolve_native(
            columns, ["x"], target, population_size=4, generations=1)
        assert pareto and all(p["kernel_path"] == "fallback" for p in pareto)
    finally:
        enforce_native_search(False)


def test_benchmark_harnesses_refuse_the_python_fallback():
    """A benchmark number must never come from the Python loop by accident.

    Every ``SearchConfig`` a harness under ``benchmarks/`` builds has to set
    ``strict_native_ga=True``; otherwise a kernel-less machine silently records
    fallback timings and error rates as native ones. Source scan on purpose:
    importing the harnesses would execute their experiments.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "benchmarks"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"SearchConfig\(", source):
            # The config literal that follows; wide enough for every harness's
            # argument block, narrow enough not to reach a neighbouring call.
            window = source[match.start(): match.start() + 600]
            if "strict_native_ga" not in window:
                line = source[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(root.parent)}:{line}")
    assert not offenders, (
        "benchmark harness builds a SearchConfig without strict_native_ga, so a "
        "missing kernel would be measured as Python: " + ", ".join(offenders))


def test_benchmark_entry_points_arm_process_wide_enforcement():
    """A harness that builds its own engines must arm the switch before running.

    ``run_synthesis_benchmark.run_benchmark`` and
    ``Synthesis50TournamentRunner.run_all`` construct the synthesis engine and
    the competitor adapters internally, and those build their own bridges: a
    ``SearchConfig`` flag cannot reach them, only the process-wide switch can.
    Source scan on purpose: importing the harnesses would execute their
    experiments.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    entry_points = {
        "benchmarks/run_synthesis_benchmark.py": "run_benchmark",
        "benchmarks/run_synthesis_50_tournament.py": "run_all",
    }
    search_calls = {
        "SearchConfig", "evolve", "RustKernelBridge",
        "AutonomousModelSynthesizer", "_run_competitor",
    }
    problems = []
    for rel, func_name in entry_points.items():
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        entry = next(
            (node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == func_name),
            None,
        )
        if entry is None:
            problems.append(f"{rel}: entry point {func_name}() not found")
            continue
        armed_at = []
        search_at = []
        for node in ast.walk(entry):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "id", None) or getattr(node.func, "attr", "")
            if called == "enforce_native_search":
                if (len(node.args) == 1
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value is True):
                    armed_at.append(node.lineno)
            elif called in search_calls:
                search_at.append(node.lineno)
        if not armed_at:
            problems.append(
                f"{rel}: {func_name}() never calls enforce_native_search(True)")
        elif search_at and min(armed_at) > min(search_at):
            problems.append(
                f"{rel}: {func_name}() arms enforcement at line {min(armed_at)}, "
                f"after its first search call at line {min(search_at)}")
    assert not problems, (
        "benchmark entry points can run unenforced: " + "; ".join(problems))


def test_strict_succeeds_when_the_kernel_is_present():
    x = np.linspace(-3, 3, 200)
    y = 3.0 * x ** 2 + 2.0
    res = evolve({"x": x}, y, config=SearchConfig(
        use_native_ga=True, strict_native_ga=True, islands=4, island_size=50,
        generations=20, seed=42))
    assert res.history[0].get("native_rust") is True
    assert min(p["train_mse"] for p in res.pareto_front) < 1e-6
