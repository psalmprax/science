"""Tests for Bootstrap Structural Stability Runner.

Verifies:
- Empirical recovery probability S(f) calculation across bootstrap trials.
- High stability detection for well-supported ground-truth structures (S(f) >= 80%).
- Ambiguity detection when multiple structures compete across resamples.
- Resampling with replacement and noise jitter.
"""
from __future__ import annotations

import numpy as np

from apedp.bootstrap_stability import (
    BootstrapStabilityReport,
    BootstrapStabilityRunner,
)


def test_bootstrap_stability_high_confidence_quadratic():
    """Synthetic dataset generated from y = 2.0*x**2 + 1.0.
    A simple polynomial regression discovery function should reliably recover quadratic."""
    runner = BootstrapStabilityRunner(default_trials=15, stability_threshold=0.70)

    rng = np.random.default_rng(101)
    xs = np.linspace(-2.0, 2.0, 50)
    ys = 2.0 * xs**2 + 1.0 + rng.normal(0, 0.05, size=len(xs))
    x_data = {"x": xs}

    def dummy_discovery(x_dict: dict[str, np.ndarray], y_sample: np.ndarray, seed: int) -> str:
        # Fits quadratic vs linear; quadratic consistently wins because curvature is strong
        x = x_dict["x"]
        deg2_err = np.mean((np.polyval(np.polyfit(x, y_sample, 2), x) - y_sample)**2)
        deg1_err = np.mean((np.polyval(np.polyfit(x, y_sample, 1), x) - y_sample)**2)
        if deg2_err < deg1_err:
            p = np.polyfit(x, y_sample, 2)
            return f"{p[0]:.3f}*x**2 + {p[1]:.3f}*x + {p[2]:.3f}"
        p = np.polyfit(x, y_sample, 1)
        return f"{p[0]:.3f}*x + {p[1]:.3f}"

    report = runner.evaluate_stability(
        x_data=x_data,
        y_data=ys,
        discovery_fn=dummy_discovery,
        n_trials=15,
        seed=42,
    )

    assert isinstance(report, BootstrapStabilityReport)
    assert report.dominant_family == "quadratic"
    assert report.stability_score >= 0.80
    assert report.is_stable is True
    assert report.n_trials == 15
    assert len(report.trials) == 15


def test_bootstrap_stability_detects_instability_under_severe_noise():
    """In a high-noise or near-linear regime, discovery alternates between linear and quadratic.
    Runner should report instability (is_stable=False)."""
    runner = BootstrapStabilityRunner(default_trials=20, stability_threshold=0.80)

    rng = np.random.default_rng(202)
    # Weak curvature, heavy noise
    xs = np.linspace(0.0, 1.0, 30)
    ys = 0.02 * xs**2 + 1.5 * xs + rng.normal(0, 0.5, size=len(xs))
    x_data = {"x": xs}

    def noisy_discovery(x_dict: dict[str, np.ndarray], y_sample: np.ndarray, seed: int) -> str:
        # Alternates based on sample noise
        r = np.random.default_rng(seed)
        if r.random() > 0.5:
            return "1.5*x + 0.1"
        return "0.05*x**2 + 1.4*x + 0.05"

    report = runner.evaluate_stability(
        x_data=x_data,
        y_data=ys,
        discovery_fn=noisy_discovery,
        n_trials=20,
        seed=100,
    )

    # Stability should fall below 0.80 threshold
    assert report.stability_score < 0.80
    assert report.is_stable is False
    assert len(report.family_counts) >= 2


def test_bootstrap_stability_handles_discovery_failure_gracefully():
    runner = BootstrapStabilityRunner(default_trials=5)
    xs = np.linspace(0.0, 1.0, 10)
    ys = xs.copy()

    def failing_discovery(x_dict, y, seed):
        if seed % 2 == 0:
            raise RuntimeError("Optimization divergence")
        return "x"

    report = runner.evaluate_stability(
        x_data={"x": xs},
        y_data=ys,
        discovery_fn=failing_discovery,
        n_trials=5,
        seed=10,
    )

    assert "FAILED_EVALUATION" in report.family_counts
    assert report.n_trials == 5


def test_ambiguity_gate_ambiguous_case_suggests_xfail():
    """Engine/harness separation: low stability + narrow gap -> AMBIGUOUS -> XFAIL.

    Mirrors the 808 finite-sample pattern without running synthesis.
    """
    from apedp.bootstrap_stability import (
        BootstrapPolicy,
        BootstrapStabilityReport,
        IdentifiabilityStatus,
        suggest_verdict,
    )

    report = BootstrapStabilityReport(
        dominant_family="rational_transition",
        stability_score=0.80,
        is_stable=True,
        stability_threshold=0.60,
        family_frequencies={"rational_transition": 0.80, "envelope": 0.20},
        family_counts={"rational_transition": 4, "envelope": 1},
        n_trials=5,
        family_stability=0.80,
        family_mean_mse={"rational_transition": 3.0e-05, "envelope": 3.2e-05},
        mse_gap=2.0e-06,
        identifiability=IdentifiabilityStatus.AMBIGUOUS.value,
        selection_margin=1.0e-04,
        reason="family_stability=0.80 (gate<=0.80); mse_gap=2.000e-06 (margin<=1.000e-04)",
    )
    assert suggest_verdict(report) == "XFAIL"
    summary = report.diagnostic_summary()
    assert "AMBIGUOUS" in summary
    assert "0.80" in summary
    d = report.to_dict()
    assert d["identifiability"] == "AMBIGUOUS"
    assert d["mse_gap"] == 2.0e-06

    policy = BootstrapPolicy()
    assert policy.min_family_stability == 0.80
    assert policy.min_trials == 10


def test_ambiguity_gate_identifiable_case_suggests_pass():
    """High stability + wide gap -> IDENTIFIABLE -> PASS."""
    from apedp.bootstrap_stability import (
        BootstrapStabilityReport,
        IdentifiabilityStatus,
        suggest_verdict,
    )

    report = BootstrapStabilityReport(
        dominant_family="power_law",
        stability_score=1.0,
        is_stable=True,
        stability_threshold=0.60,
        family_frequencies={"power_law": 1.0},
        family_counts={"power_law": 10},
        n_trials=10,
        family_stability=1.0,
        family_mean_mse={"power_law": 9.0e-06},
        mse_gap=None,
        identifiability=IdentifiabilityStatus.IDENTIFIABLE.value,
        selection_margin=1.0e-04,
        reason="family_stability=1.00 (gate<=0.80)",
    )
    assert suggest_verdict(report) == "PASS"


def test_ambiguity_gate_insufficient_case_suggests_fail():
    """Unusable trials -> INSUFFICIENT -> FAIL (not XFAIL)."""
    from apedp.bootstrap_stability import (
        BootstrapStabilityReport,
        IdentifiabilityStatus,
        suggest_verdict,
    )

    report = BootstrapStabilityReport(
        dominant_family="SYNTHESIS_ERROR",
        stability_score=1.0,
        is_stable=False,
        stability_threshold=0.60,
        family_frequencies={"SYNTHESIS_ERROR": 1.0},
        family_counts={"SYNTHESIS_ERROR": 5},
        n_trials=5,
        family_stability=1.0,
        family_mean_mse={},
        mse_gap=None,
        identifiability=IdentifiabilityStatus.INSUFFICIENT.value,
        selection_margin=None,
        reason="cannot measure",
    )
    assert suggest_verdict(report) == "FAIL"
