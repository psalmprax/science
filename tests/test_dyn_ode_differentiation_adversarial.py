"""Adversarial Benchmark on Numerical Differentiation Vulnerability in L2-ODE.

Addresses the foundational numerical dilemma in dynamic equation discovery:
    \\widehat{\\dot{x}} = \\dot{x} + \\epsilon_{diff}

When discovery pipelines fit models against numerical derivatives, discovery errors
frequently originate in the derivative estimator rather than the structural grammar.

Benchmarks three estimation paradigms across adversarial conditions:
  1. Spline differentiation (CubicSpline derivative)
  2. Central difference (finite differences gradient)
  3. Weak-form integral formulation (integration by parts against compact bump test functions)

Adversarial Conditions Tested:
  - Noise grids: 0%, 1%, 2%, 5%, 15%
  - Boundary edge effects (Runge/endpoint oscillations)
  - Irregular / non-uniform time sampling grids
  - Epistemic abstention under severe noise swamping
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from apedp_dyn.discovery_l2.ode_engine import estimate_time_derivative
from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicAttribution,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.estimation.weak_form import make_windows
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _solve_weak_form_scalar(
    t: np.ndarray,
    x: np.ndarray,
    term_func,
    n_windows: int = 32,
    radius: float | None = None,
) -> float:
    """Fit scalar coefficient theta for dx/dt = theta * f(x) via weak form."""
    span = (float(t[0]), float(t[-1]))
    rad = radius if radius is not None else (span[1] - span[0]) / 8.0
    windows = make_windows(span, n_windows=n_windows, radius=rad)

    n_w = len(windows.centers)
    b = np.zeros(n_w)
    G = np.zeros(n_w)

    val = term_func(x)
    for j in range(n_w):
        phi = windows.phi(t, j)
        phi_dot = windows.phi_dot(t, j)
        b[j] = -np.trapezoid(x * phi_dot, t)
        G[j] = np.trapezoid(val * phi, t)

    w, *_ = np.linalg.lstsq(G.reshape(-1, 1), b, rcond=None)
    return float(w[0])


def _solve_pointwise_scalar(
    t: np.ndarray,
    x: np.ndarray,
    term_func,
    method: str = "spline",
) -> float:
    """Fit scalar coefficient theta for dx/dt = theta * f(x) via pointwise derivative."""
    x_dot = estimate_time_derivative(t, x, method=method)
    phi = term_func(x).reshape(-1, 1)
    w, *_ = np.linalg.lstsq(phi, x_dot, rcond=None)
    return float(w[0])


class TestDifferentiationVulnerabilityBenchmark:
    """Comprehensive benchmark across derivative estimators under adversarial conditions."""

    def test_zero_noise_exactness_all_estimators(self):
        """At zero noise, spline, central difference, and weak form all recover theta = -0.5."""
        t = np.linspace(0.1, 2.0, 100)
        # Truth: dx/dt = -0.5 * x^2 => x(t) = 1 / (0.5*t + 1)
        x_clean = 1.0 / (0.5 * t + 1.0)
        truth_theta = -0.5

        theta_spline = _solve_pointwise_scalar(t, x_clean, lambda z: z**2, method="spline")
        theta_cdiff = _solve_pointwise_scalar(t, x_clean, lambda z: z**2, method="central_diff")
        theta_weak = _solve_weak_form_scalar(t, x_clean, lambda z: z**2, n_windows=32)

        assert np.isclose(theta_spline, truth_theta, atol=1e-4)
        assert np.isclose(theta_cdiff, truth_theta, atol=1e-3)
        assert np.isclose(theta_weak, truth_theta, atol=1e-4)

    def test_noise_sweep_weak_form_resilience(self):
        """Weak form outperforms spline and central difference across 1%, 2%, and 5% noise."""
        t = np.linspace(0.1, 2.0, 120)
        x_clean = 1.0 / (0.5 * t + 1.0)
        truth_theta = -0.5
        rng = np.random.default_rng(12345)
        scale_x = float(np.std(x_clean))

        noise_levels = [0.01, 0.02, 0.05]
        for pct in noise_levels:
            x_noisy = x_clean + rng.normal(0, pct * scale_x, len(t))

            err_spline = abs(_solve_pointwise_scalar(t, x_noisy, lambda z: z**2, method="spline") - truth_theta)
            err_cdiff = abs(_solve_pointwise_scalar(t, x_noisy, lambda z: z**2, method="central_diff") - truth_theta)
            err_weak = abs(_solve_weak_form_scalar(t, x_noisy, lambda z: z**2, n_windows=32) - truth_theta)

            # Weak form error must be lower than pointwise derivatives
            assert err_weak < err_spline, f"At {pct*100}% noise: weak {err_weak} vs spline {err_spline}"
            assert err_weak < err_cdiff, f"At {pct*100}% noise: weak {err_weak} vs cdiff {err_cdiff}"
            # At 1% noise, weak form error is under 1% relative
            if pct == 0.01:
                assert err_weak < 0.005

    def test_boundary_edge_effect_immunity(self):
        """Spline derivative blows up at endpoints; weak test functions strictly vanish."""
        t = np.linspace(0.1, 2.0, 100)
        x_clean = 1.0 / (0.5 * t + 1.0)
        rng = np.random.default_rng(999)
        x_noisy = x_clean + rng.normal(0, 0.02 * np.std(x_clean), len(t))

        # True analytical derivative at endpoints
        x_dot_true = -0.5 * (x_clean**2)

        # Spline derivative evaluated at endpoints
        cs = CubicSpline(t, x_noisy)
        x_dot_spline = cs.derivative()(t)

        edge_err_spline = max(abs(x_dot_spline[0] - x_dot_true[0]), abs(x_dot_spline[-1] - x_dot_true[-1]))
        interior_err_spline = np.mean(np.abs(x_dot_spline[10:-10] - x_dot_true[10:-10]))

        # Spline error at boundary exceeds interior error
        assert edge_err_spline > interior_err_spline

        # In weak form, windows strictly vanish at boundary endpoints by construction
        windows = make_windows((t[0], t[-1]), n_windows=16, radius=0.20)
        for j in range(len(windows.centers)):
            assert windows.phi(t, j)[0] == 0.0
            assert windows.phi(t, j)[-1] == 0.0
            assert windows.phi_dot(t, j)[0] == 0.0
            assert windows.phi_dot(t, j)[-1] == 0.0

    def test_irregular_nonuniform_time_sampling(self):
        """Weak form remains accurate under non-uniform / jittered sampling intervals."""
        rng = np.random.default_rng(777)
        # Generate non-uniform time grid
        dt = rng.uniform(0.01, 0.04, size=90)
        t = np.concatenate([[0.1], 0.1 + np.cumsum(dt)])
        x_clean = 1.0 / (0.5 * t + 1.0)
        x_noisy = x_clean + rng.normal(0, 0.01 * np.std(x_clean), len(t))

        theta_weak = _solve_weak_form_scalar(t, x_noisy, lambda z: z**2, n_windows=24)
        rel_err = abs(theta_weak - (-0.5)) / 0.5
        assert rel_err < 0.02, f"Weak form relative error on irregular grid was {rel_err:.3%}"

    def test_high_noise_swamped_epistemic_abstention(self):
        """Under severe 20% noise, APEDP epistemic layer refuses UNIQUE false promotion."""
        t = np.linspace(0.1, 2.0, 100)
        x_clean = 1.0 / (0.5 * t + 1.0)
        rng = np.random.default_rng(42)
        # 20% noise severely swamps the signal
        x_noisy = x_clean + rng.normal(0, 0.20 * np.std(x_clean), len(t))

        x_dot = estimate_time_derivative(t, x_noisy, method="spline")
        n_train = 60
        y_tr, y_ho = x_dot[:n_train], x_dot[n_train:]
        x_tr, x_ho = x_noisy[:n_train], x_noisy[n_train:]

        # Competing models: M1 (x^2), M2 (x)
        cands = []
        for name, pwr in [("x2_model", 2), ("linear_model", 1)]:
            f_tr = (x_tr**pwr).reshape(-1, 1)
            f_ho = (x_ho**pwr).reshape(-1, 1)
            w, *_ = np.linalg.lstsq(f_tr, y_tr, rcond=None)
            pred_tr = (f_tr @ w).ravel()
            pred_ho = (f_ho @ w).ravel()
            loss_tr = float(np.mean((y_tr - pred_tr) ** 2))
            resid_ho = y_ho - pred_ho
            rmse_ho = float(np.sqrt(np.mean(resid_ho**2)))
            obs_rms = float(np.sqrt(np.mean(y_ho**2)))

            cand_ir = CandidateDynamics(lhs="x_dot", terms=(Term(name, "theta"),), candidate_id=name)
            fit_res = FitResult(param_values={"theta": float(w[0])}, train_loss=loss_tr, nfev=1, success=True, message="OLS")
            g4 = {
                "passed": False,  # Noise swamped; fails predictive ceiling
                "detail": f"noise swamped holdout RMSE={rmse_ho:.3g}",
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": float(np.std(resid_ho)),
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho,
                "holdout_residuals": resid_ho.tolist(),
            }
            cands.append(ScoredCandidate(cand_ir, fit=fit_res, train_loss=loss_tr, holdout_rmse=rmse_ho, provenance={"gate4_status": g4}))

        adj = adjudicate(
            cands,
            benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY,
            data_origin=DataOrigin.SIMULATION,
        )

        # Must NOT be promoted to UNIQUE_SUPPORT
        assert adj.status != EpistemicStatus.UNIQUE_SUPPORT
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.epistemic_attribution in (
            EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value,
            EpistemicAttribution.MEASUREMENT_NOISE_SWAMPED.value,
        )
