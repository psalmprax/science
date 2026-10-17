"""Observability + cache regression tests for the Rust bridge.

Three behaviors pin this file's design:

1. **Path counters.** A run must be able to *prove* which engine answered —
   `bridge.tallies` and the `kernel_path` field attached to result dicts. This
   is the fix the bridge's own audit comment asked for ("a counter, not a log
   line"): a benchmark that meant to measure the Rust fast path cannot
   silently measure the Python reference engine.

2. **Compile cache.** `kernel.compile` is a pure function of
   ``(expr_str, variable_names)``, so repeated candidates must not re-parse at
   every FFI crossing. The cache key includes the variable table, so an
   expression served against a different column layout cannot reuse a
   compiled program whose variable indexing depends on that layout.

3. **Loud missing-kernel warning.** Constructions without the native kernel
   warn once at WARNING level — not at debug — so a benchmark run's log
   actually shows the fallback instead of burying it.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import apedp.rust_bridge as rb
from apedp.rust_bridge import RustKernelBridge


@pytest.fixture
def fake_kernel(monkeypatch):
    """Same shape as the dispatch-test kernel: records compile calls."""

    class RecordingKernel:
        def __init__(self) -> None:
            self.compile_calls: list[tuple[str, tuple]] = []

        def compile(self, expr_str, variable_names):
            self.compile_calls.append((expr_str, tuple(variable_names)))
            return ("compiled", expr_str)

        def evaluate_batch(self, compiled, data):
            return np.full(len(data[0]), 7.0)

        def compute_metrics(self, yt, yp):
            return (1.5, 0.5, 0.9, 2.5, False)

        def score_candidates_parallel(self, compiled, data, target):
            return [(i, 0.0, 0.0, 1.0, 0.0, False) for i in range(len(compiled))]

        def evolve_native(self, *args, **kwargs):
            return ([("x0", 0.1, 1, 1.0, 0.0)], 1)

    recording = RecordingKernel()
    monkeypatch.setattr(rb, "apedp_kernel", recording)
    monkeypatch.setattr(rb, "_RUST_AVAILABLE", True)
    return recording


def test_tallies_prove_the_fast_path_ran(fake_kernel) -> None:
    """After kernel-backed calls, the tally must say `rust` — not stay silent."""
    bridge = RustKernelBridge()
    x = np.linspace(0.1, 2.0, 5)
    bridge.evaluate_expression("x0**2", ["x0"], [x])
    out = bridge.compute_metrics(x, x)

    assert bridge.tallies["rust"] >= 2, (
        f"kernel-backed calls did not increment the rust tally: {bridge.tallies}")
    assert bridge.tallies["fallback"] == 0
    assert bridge.last_path == "rust"
    assert out["kernel_path"] == "rust"


def test_result_dicts_carry_kernel_path(fake_kernel) -> None:
    bridge = RustKernelBridge()
    x = np.linspace(0.1, 2.0, 5)
    rows = bridge.score_candidates_batch(["x0"], ["x0"], [x], x)
    assert rows, "batch scoring returned nothing"
    for row in rows:
        assert row["kernel_path"] == "rust", row


def test_failing_kernel_tallies_fallback(fake_kernel, monkeypatch) -> None:
    """A raising entry point must record itself as a fallback, not vanish."""

    class FailingKernel:
        def compile(self, *a, **k):
            raise RuntimeError("simulated kernel failure")

        def compute_metrics(self, yt, yp):
            raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr(rb, "apedp_kernel", FailingKernel())
    bridge = RustKernelBridge()
    x = np.linspace(0.1, 3.0, 7)
    out = bridge.compute_metrics(x, x + 0.1)
    assert out["kernel_path"] == "fallback"
    assert bridge.tallies["fallback"] == 1
    assert bridge.tallies["rust"] == 0


def test_compile_cache_serves_repeated_expressions_once(fake_kernel) -> None:
    """Two identical crossings must reuse the compiled program's handle."""
    bridge = RustKernelBridge()
    x = np.linspace(0.1, 2.0, 5)
    bridge.evaluate_expression("x0**2 + x0", ["x0"], [x])
    bridge.evaluate_expression("x0**2 + x0", ["x0"], [x])
    bridge.evaluate_expression("x0**2 + x0", ["x0"], [x])
    compile_calls = [c for c in fake_kernel.compile_calls
                     if c[0] == "x0**2 + x0"]
    assert len(compile_calls) == 1, (
        f"the compile cache was not consulted: {fake_kernel.compile_calls}")


def test_compile_cache_is_keyed_on_the_variable_table(fake_kernel) -> None:
    """The same text under a different variable table must re-compile.

    The compiled program indexes columns by the declared table order, so
    caching only on the expression string would evaluate against the wrong
    columns.
    """
    bridge = RustKernelBridge()
    x = np.linspace(0.1, 2.0, 5)
    bridge.evaluate_expression("x0", ["x0"], [x])
    bridge.evaluate_expression("x0", ["x0", "x1"], [x, x])
    # NOTE: upstream protocol requires len(data) == num_variables, so a second
    # table is supplied its own column count; both calls are dispatched.
    tables = {c for c in fake_kernel.compile_calls if c[0] == "x0"}
    assert ("x0", ("x0",)) in tables
    assert ("x0", ("x0", "x1")) in tables, (
        f"the cache key ignored the variable table: {fake_kernel.compile_calls}")


def test_fallback_evaluates_from_the_lambdify_cache() -> None:
    """The Python reference engine must not re-lambdify identical expressions."""
    bridge = RustKernelBridge(force_fallback=True)
    x = np.linspace(0.1, 3.0, 7)
    first = bridge._fallback_eval("x0**2 + 1.0", ["x0"], [x])
    size_after_first = len(rb._lambdify_cache)

    second = bridge._fallback_eval("x0**2 + 1.0", ["x0"], [x])
    assert len(rb._lambdify_cache) == size_after_first, (
        "lambdify compiled the same expression twice")
    assert np.allclose(first, second)
    assert np.allclose(first, x**2 + 1.0)


def test_missing_kernel_warns_once_loudly(monkeypatch, caplog) -> None:
    """Constructions without the kernel must log WARNING, and only once."""
    monkeypatch.setattr(rb, "_RUST_AVAILABLE", False)
    monkeypatch.setattr(rb, "_missing_kernel_warned", False)
    with caplog.at_level(logging.DEBUG, logger="apedp.rust_bridge"):
        RustKernelBridge()
        RustKernelBridge()
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "not built" in r.message]
    assert len(warnings) == 1, (
        "the missing-kernel condition must warn exactly once, at WARNING level")
    assert "maturin develop" in warnings[0].message, (
        "the warning must name the build command, not merely complain")


def test_force_fallback_does_not_warn_about_the_kernel(
    fake_kernel, monkeypatch, caplog
) -> None:
    """force_fallback is a deliberate choice; warning about it would be noise."""
    monkeypatch.setattr(rb, "_missing_kernel_warned", False)
    with caplog.at_level(logging.WARNING, logger="apedp.rust_bridge"):
        RustKernelBridge(force_fallback=True)
    assert not [r for r in caplog.records
                if r.levelno == logging.WARNING and "not built" in r.message]
