"""Test the Rust bridge's *dispatch* with a fake kernel, and declare the gap.

What is untested, and why it stayed that way
--------------------------------------------
``apedp/rust_bridge.py`` has eight ``if self.use_rust:`` branches -- one per
kernel-backed method. Branch coverage put the module at 20 of 64 with all eight
untaken, and they are untaken in *every* environment:

- ``apedp_kernel`` is a native extension and is not built in this checkout
  (``import apedp_kernel`` raises ``ModuleNotFoundError``), so ``_RUST_AVAILABLE``
  is ``False`` and ``use_rust`` can never become ``True`` here;
- ``.github/workflows/test.yml`` contains no ``cargo``, ``maturin`` or
  ``rust`` step at all, so CI does not build it either;
- ``tests/test_cross_implementation_conformance.py`` -- the suite that would
  cover those branches, via differential fuzzing of Rust against Python --
  skips on the missing module. Roughly seventeen skipped tests.

So the only rust_bridge tests that actually run
(``test_rust_bridge_fallback_safety.py``) cover the Python fallback, and the
Rust half of the bridge is verified nowhere. That is the dangerous shape: the
branch is a *dispatch* layer, where a wrong argument order or a wrong tuple
unpack produces a plausible wrong number rather than a crash, and the code that
would notice is skipped.

What these tests do instead
--------------------------
They exercise the bridge with a **fake kernel** injected by monkeypatching
``apedp.rust_bridge.apedp_kernel``. That tests the layer this file is about --
does the bridge call the kernel correctly, and does it degrade to Python when
the kernel fails -- which is testable with no Rust toolchain.

What remains unverifiable here is stated in the last test rather than left
implicit: the *kernel's own numerics*. A fake kernel cannot tell you whether
Rust and Python agree numerically, because that is the one question only the
real kernel can answer. ``test_the_conformance_suite_is_not_silently_skipped``
exists so that gap is reported rather than forgotten.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import apedp.rust_bridge as rb
from apedp.rust_bridge import RustKernelBridge, is_rust_kernel_available


class FakeKernel:
    """Stands in for the native extension, recording how it was called.

    ``fails`` makes every entry point raise, which is how the bridge's
    degradation path gets exercised.
    """

    def __init__(self, *, fails: bool = False) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.fails = fails

    def _record(self, name: str, *args):
        self.calls.append((name, args))
        if self.fails:
            raise RuntimeError(f"simulated kernel failure in {name}")

    def compile(self, expr_str, variable_names):
        self._record("compile", expr_str, tuple(variable_names))
        return ("compiled", expr_str)

    def evaluate_batch(self, compiled, data):
        self._record("evaluate_batch", compiled, tuple(len(d) for d in data))
        return np.full(len(data[0]), 7.0)

    def compute_metrics(self, yt, yp):
        self._record("compute_metrics", np.asarray(yt).shape, np.asarray(yp).shape)
        return (1.5, 0.5, 0.9, 2.5, False)

    def score_candidates_parallel(self, compiled, data, target):
        self._record("score_candidates_parallel", len(compiled), len(target))
        # The real kernel returns one row per candidate: (id, mse, mae, r2,
        # max_res, has_nan). Returning the wrong shape here is the kind of
        # bug this file exists to catch.
        return [(i, 0.0, 0.0, 1.0, 0.0, False) for i in range(len(compiled))]

    def probe_singularity(self, compiled, bounds, num_grid_points):
        self._record("probe_singularity", compiled, tuple(bounds), num_grid_points)
        return (False, False, None, 0.0, "OK")

    def compute_matrix_conditioning(self, cols, rcond_threshold):
        self._record("compute_matrix_conditioning", len(cols), rcond_threshold)
        return (2.0, 3.0, 0.0, 3, True, [0.1, 0.2, 0.3])

    def solve_linear_coefficients(self, cols, target, lambda_reg):
        self._record("solve_linear_coefficients", len(cols), lambda_reg)
        return ([1.0, 2.0], 0.1, 0.05, 0.9, 2.0, True)

    def evaluate_precision_boundary(self, delta, cond, bits):
        self._record("evaluate_precision_boundary", delta, cond, bits)
        return ("SAFE", 0.25, 1.0, False, True)


@pytest.fixture
def fake_kernel(monkeypatch):
    """Install a fake kernel and make ``_RUST_AVAILABLE`` true.

    ``force_fallback=False`` on its own would not be enough: ``use_rust`` is
    ``_RUST_AVAILABLE and not force_fallback``, so the module-level flag has to
    be patched too or the bridge can never select the kernel no matter what the
    constructor says.
    """
    kernel = FakeKernel()
    monkeypatch.setattr(rb, "apedp_kernel", kernel)
    monkeypatch.setattr(rb, "_RUST_AVAILABLE", True)
    return kernel


@pytest.fixture
def failing_kernel(monkeypatch):
    kernel = FakeKernel(fails=True)
    monkeypatch.setattr(rb, "apedp_kernel", kernel)
    monkeypatch.setattr(rb, "_RUST_AVAILABLE", True)
    return kernel


class TestKernelAvailabilityIsHonest:
    def test_force_fallback_overrides_availability(self, fake_kernel) -> None:
        """The escape hatch must actually win, or it is not one."""
        bridge = RustKernelBridge(force_fallback=True)
        assert bridge.use_rust is False, (
            "force_fallback=True but use_rust is True; the flag is ignored when "
            "the kernel happens to be built")

    def test_default_uses_the_kernel_when_available(self, fake_kernel) -> None:
        assert RustKernelBridge().use_rust is True

    def test_the_availability_probe_follows_the_module_flag(self, monkeypatch) -> None:
        monkeypatch.setattr(rb, "_RUST_AVAILABLE", True)
        assert is_rust_kernel_available() is True
        monkeypatch.setattr(rb, "_RUST_AVAILABLE", False)
        assert is_rust_kernel_available() is False


class TestDispatchReachesTheKernel:
    """Every method with an ``if self.use_rust:`` branch, on the kernel side."""

    def test_evaluate_expression_compiles_then_evaluates(self, fake_kernel) -> None:
        x = np.linspace(0.1, 2.0, 5)
        out = RustKernelBridge().evaluate_expression("x0**2", ["x0"], [x])
        assert [c[0] for c in fake_kernel.calls] == ["compile", "evaluate_batch"]
        assert np.allclose(out, 7.0), "the kernel's result was not returned"

    def test_compute_metrics_unpacks_the_kernel_tuple(self, fake_kernel) -> None:
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([1.1, 2.1, 2.9])
        out = RustKernelBridge().compute_metrics(yt, yp)
        assert out["mse"] == 1.5 and out["max_residual"] == 2.5
        assert "has_nan_or_inf" in out, (
            f"the kernel's nan flag was dropped; keys are {sorted(out)}")

    def test_every_kernel_backed_method_dispatches(self, fake_kernel) -> None:
        """The guard against a new branch being added without a test.

        Each of the eight ``if self.use_rust:`` sites is exercised here. If a
        ninth is added later this test does not fail -- but the audit that
        produced this file will report it as untaken, and that combination is
        the point.
        """
        bridge = RustKernelBridge()
        x = np.linspace(0.1, 2.0, 4)
        bridge.evaluate_expression("x0", ["x0"], [x])
        bridge.compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
        bridge.score_candidates_batch(["x0", "x0**2"], ["x0"], [x], x)
        # Public methods take (expr, variable_names, bounds, n) -- four
        # args -- while the _RustKernel Protocol methods take the already-
        # compiled form. Conflating the two layers produced a bounds
        # argument of 8 and a TypeError deep in the fallback.
        bridge.probe_singularity("x0", ["x0"], [(0.0, 1.0)], 8)
        bridge.compute_matrix_conditioning([x, x**2], 1e-12)
        bridge.solve_linear_coefficients([x, x**2], np.array([1.0, 2.0]), 1e-9)
        bridge.evaluate_precision_boundary(1e-3, 5.0, 32)
        called = {c[0] for c in fake_kernel.calls}
        for expected in ("compile", "evaluate_batch", "compute_metrics",
                         "score_candidates_parallel", "probe_singularity",
                         "compute_matrix_conditioning",
                         "solve_linear_coefficients",
                         "evaluate_precision_boundary"):
            assert expected in called, f"{expected} was never reached on the Rust path"


class TestKernelFailureDegradesToPython:
    """The property that actually matters: a broken kernel must not be trusted."""

    def test_a_raising_kernel_still_returns_the_right_numbers(self, failing_kernel,
                                                              monkeypatch) -> None:
        """Not merely "does not crash" -- the answer must be correct.

        A bridge that returns zeros on kernel failure would pass every
        "did it raise?" test and be silently wrong. So this compares against the
        same expression evaluated through the Python-only bridge.
        """
        x = np.linspace(0.1, 3.0, 7)
        expr = "x0**2 + 1.0"
        degraded = RustKernelBridge().evaluate_expression(expr, ["x0"], [x])
        expected = RustKernelBridge(force_fallback=True).evaluate_expression(
            expr, ["x0"], [x])
        assert np.allclose(degraded, expected, equal_nan=True), (
            "a failing kernel produced different numbers from the Python "
            "fallback; degradation changed the answer")
        assert [c[0] for c in failing_kernel.calls] == ["compile"], (
            "the kernel was not even tried, so this did not test degradation")

    def test_metrics_degrade_to_the_python_computation(self, failing_kernel) -> None:
        yt = np.array([1.0, 2.0, 3.0, 4.0])
        yp = np.array([1.2, 1.9, 3.3, 3.7])
        out = RustKernelBridge().compute_metrics(yt, yp)
        expected = RustKernelBridge(force_fallback=True).compute_metrics(yt, yp)
        assert out == expected, (
            f"degraded metrics {out} differ from the Python path {expected}")

    def test_dual_validation_logs_a_divergence(self, fake_kernel, caplog) -> None:
        """``if diff > 1e-10:`` -- untaken, and it is the whole safety net.

        The fake returns a constant 7.0 while Python evaluates the expression,
        so the two must disagree and the bridge must say so rather than
        preferring the kernel silently.
        """
        import logging

        x = np.linspace(0.1, 3.0, 5)
        with caplog.at_level(logging.WARNING, logger="apedp.rust_bridge"):
            RustKernelBridge(dual_validate=True).evaluate_expression(
                "x0**2", ["x0"], [x])
        assert any("Dual-validation divergence" in r.message for r in caplog.records), (
            "the kernel and Python disagreed by far more than 1e-10 and nothing "
            "was logged; dual_validate is decorative")


class TestTheRemainingGapIsDeclared:
    def test_the_conformance_suite_is_not_silently_skipped(self) -> None:
        """State the gap rather than leaving it to be rediscovered.

        If someone builds ``apedp_kernel``, this file's fake-kernel tests still
        pass and the real differential fuzzing becomes possible -- so the check
        below should flip to demanding it runs.
        """
        conformance = (Path(__file__).resolve().parent
                       / "test_cross_implementation_conformance.py")
        assert conformance.is_file(), (
            "the Rust/Python conformance suite has been removed; if that was "
            "deliberate, the eight rust_bridge branches now have no differential "
            "coverage anywhere")
        if is_rust_kernel_available():
            return  # kernel present: the suite runs, nothing to declare

        assert "apedp_kernel native extension not built" in conformance.read_text(), (
            "the conformance suite is expected to skip here; if it was changed "
            "to skip for another reason, say so in this file rather than "
            "letting the reason go unstated")