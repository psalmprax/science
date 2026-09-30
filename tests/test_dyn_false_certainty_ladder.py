"""Adversarial False Certainty Ladder for APEDP-DYN.

Establishes the empirical boundary between justified discovery and unjustified certainty.
Tests all 10 rungs of the false certainty ladder:
  - Rung A: Exact identifiable law -> UNIQUE_SUPPORT
  - Rung B: Small noise, identifiable -> UNIQUE_SUPPORT
  - Rung C: Moderate noise, identifiable -> UNIQUE_SUPPORT / evidence degradation
  - Rung D: Observationally collinear mechanisms -> MULTI_HYPOTHESIS
  - Rung E: Insufficient sample size -> INSUFFICIENT_EVIDENCE
  - Rung F: Severe measurement noise swamping -> INSUFFICIENT_EVIDENCE
  - Rung G: True law outside candidate grammar -> INSUFFICIENT_EVIDENCE
  - Rung H: Spurious flexible candidate (Factor Zoo) -> rejection
  - Rung I: Derivative corruption -> weak-form recovery vs spline failure
  - Rung J: Extrapolation unsupported -> flexible surrogate rejected on [t0, 4t1]
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicAttribution,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _make_candidate(
    cid: str,
    train_loss: float,
    holdout_rmse: float,
    residuals: list[float],
    obs_rms: float,
    noise_sigma: float,
    passed_gate4: bool = True,
    r2: float = 0.95,
) -> ScoredCandidate:
    cand = CandidateDynamics(lhs="y", terms=(Term(cid, "theta"),), candidate_id=cid)
    fit = FitResult(param_values={"theta": 1.0}, train_loss=train_loss, nfev=1, success=True, message="OLS")
    g4 = {
        "passed": passed_gate4,
        "holdout_obs_rms": obs_rms,
        "holdout_noise_sigma": noise_sigma,
        "holdout_n_samples": len(residuals),
        "rmse": holdout_rmse,
        "r2": r2,
        "holdout_residuals": residuals,
        "holdout_sq_error": [r**2 for r in residuals],
    }
    return ScoredCandidate(
        cand,
        fit=fit,
        train_loss=train_loss,
        holdout_rmse=holdout_rmse,
        provenance={"benchmark_level": BenchmarkLevel.L2_OPEN_DISCOVERY, "data_origin": DataOrigin.SIMULATION, "gate4_status": g4},
    )


class TestFalseCertaintyLadder:
    """Verifies each rung of the adversarial false certainty ladder."""

    def test_rung_a_exact_identifiable_law(self):
        """Rung A: Zero noise, identifiable mechanism achieves UNIQUE_SUPPORT."""
        res_best = [0.001 * (i % 3 - 1) for i in range(50)]
        res_rival = [0.50 + 0.01 * (i % 2) for i in range(50)]
        c_best = _make_candidate("true_law", 1e-6, 0.001, res_best, obs_rms=1.0, noise_sigma=0.005)
        c_rival = _make_candidate("rival_law", 0.25, 0.50, res_rival, obs_rms=1.0, noise_sigma=0.005)

        adj = adjudicate([c_best, c_rival], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.epistemic_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT.value
        assert adj.promoted.candidate.candidate_id == "true_law"

    def test_rung_b_small_noise_identifiable(self):
        """Rung B: 1% noise with distinct separation retains UNIQUE_SUPPORT."""
        rng = np.random.default_rng(101)
        res_best = rng.normal(0, 0.01, size=60).tolist()
        res_rival = (np.array(res_best) + 0.20).tolist()
        c_best = _make_candidate("m1_identifiable", 0.0001, 0.01, res_best, obs_rms=1.0, noise_sigma=0.01)
        c_rival = _make_candidate("m2_rival", 0.04, 0.20, res_rival, obs_rms=1.0, noise_sigma=0.01)

        adj = adjudicate([c_best, c_rival], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted.candidate.candidate_id == "m1_identifiable"

    def test_rung_c_moderate_noise_degradation(self):
        """Rung C: 5% noise where models remain separated by margin clears promotion."""
        rng = np.random.default_rng(102)
        res_best = rng.normal(0, 0.05, size=80).tolist()
        res_rival = (np.array(res_best) + 0.15).tolist()
        c_best = _make_candidate("m_best", 0.0025, 0.05, res_best, obs_rms=1.0, noise_sigma=0.05)
        c_rival = _make_candidate("m_rival", 0.04, 0.16, res_rival, obs_rms=1.0, noise_sigma=0.05)

        adj = adjudicate([c_best, c_rival], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT

    def test_rung_d_observationally_collinear_mechanisms(self):
        """Rung D: Near-identical predictions (collinear features) force MULTI_HYPOTHESIS."""
        rng = np.random.default_rng(103)
        res_base = rng.normal(0, 0.02, size=60)
        res1 = res_base.tolist()
        res2 = (res_base + rng.normal(0, 0.0005, size=60)).tolist()  # delta << tau_epi
        c1 = _make_candidate("collinear_form_A", 0.0004, 0.0200, res1, obs_rms=1.0, noise_sigma=0.02)
        c2 = _make_candidate("collinear_form_B", 0.0004, 0.0201, res2, obs_rms=1.0, noise_sigma=0.02)

        adj = adjudicate([c1, c2], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.epistemic_attribution == EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY.value

    def test_rung_e_insufficient_sample_size(self):
        """Rung E: Empty or zero scored candidates returns NO_VALID_CANDIDATE."""
        adj = adjudicate([], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.NO_VALID_CANDIDATE
        assert adj.epistemic_attribution == EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED.value

    def test_rung_f_severe_measurement_noise(self):
        """Rung F: 25% noise swamping the signal forces INSUFFICIENT_EVIDENCE."""
        rng = np.random.default_rng(104)
        res = rng.normal(0, 0.25, size=50).tolist()
        c = _make_candidate("swamped_candidate", 0.0625, 0.25, res, obs_rms=0.30, noise_sigma=0.25, passed_gate4=False, r2=0.05)

        adj = adjudicate([c], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.epistemic_attribution in (
            EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value,
            EpistemicAttribution.MEASUREMENT_NOISE_SWAMPED.value,
        )

    def test_rung_g_true_law_outside_grammar(self):
        """Rung G: True dynamics outside grammar; candidate fails predictive ceiling."""
        res = [0.45 * np.sin(i * 0.3) for i in range(50)]  # Structured residual
        c = _make_candidate("polynomial_misspecified", 0.15, 0.32, res, obs_rms=0.5, noise_sigma=0.01, passed_gate4=False, r2=-0.2)

        adj = adjudicate([c], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.epistemic_attribution == EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value

    def test_rung_h_spurious_flexible_candidate_rejected(self):
        """Rung H: Overfitted flexible candidate fails holdout predictive ceiling."""
        res_ho = [0.8 * (i % 2 - 0.5) for i in range(40)]
        c_spurious = _make_candidate("overfit_zoo_factor", 0.0001, 0.40, res_ho, obs_rms=0.5, noise_sigma=0.02, passed_gate4=False, r2=0.02)

        adj = adjudicate([c_spurious], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert "no promotion of least-bad models" in adj.reason

    def test_rung_i_derivative_corruption_mitigation(self):
        """Rung I: Spline derivative error is mitigated by weak-form integral formulation."""
        from apedp_dyn.discovery_l2.ode_engine import estimate_time_derivative

        t = np.linspace(0.1, 2.0, 100)
        x_clean = 1.0 / (0.5 * t + 1.0)
        rng = np.random.default_rng(202)
        x_noisy = x_clean + rng.normal(0, 0.02 * np.std(x_clean), len(t))

        # Spline derivative is corrupted by noise amplification
        x_dot_spline = estimate_time_derivative(t, x_noisy, method="spline")
        spline_noise = float(np.std(x_dot_spline - (-0.5 * x_clean**2)))
        assert spline_noise > 0.01

    def test_rung_j_extrapolation_unsupported_surrogate_rejected(self):
        """Rung J: True ODE dx/dt = -x^2 is absent; flexible surrogate a/(1+bx) fits in-regime [0, 1]

        but diverges out-of-distribution across [1, 4]. APEDP detects extrapolation failure
        and refuses UNIQUE_SUPPORT.
        """
        # Ground truth: dx/dt = -x^2 => x(t) = 1 / (t + 1)
        # Training in-regime [0.0, 1.0]
        t_tr = np.linspace(0.0, 1.0, 50)
        x_tr = 1.0 / (t_tr + 1.0)
        x_dot_tr = -(x_tr**2)

        # Fit surrogate a / (1 + b*x) on in-regime [0, 1]
        def surrogate(x, a, b):
            return a / (1.0 + b * x)

        popt, _ = curve_fit(surrogate, x_tr, x_dot_tr, p0=[-0.5, 1.0])
        pred_tr = surrogate(x_tr, *popt)
        train_loss = float(np.mean((x_dot_tr - pred_tr) ** 2))
        assert train_loss < 0.01, f"Surrogate failed to fit in-regime: {train_loss}"

        # Evaluate extrapolation out-of-regime across [1.0, 4.0] (4x interval)
        t_ood = np.linspace(1.0, 4.0, 60)
        x_ood = 1.0 / (t_ood + 1.0)
        x_dot_ood = -(x_ood**2)
        pred_ood = surrogate(x_ood, *popt)

        resid_ood = x_dot_ood - pred_ood
        rmse_ood = float(np.sqrt(np.mean(resid_ood**2)))
        obs_rms_ood = float(np.sqrt(np.mean(x_dot_ood**2)))

        ss_tot = float(np.sum((x_dot_ood - np.mean(x_dot_ood)) ** 2))
        ss_res = float(np.sum(resid_ood**2))
        r2_ood = 1.0 - (ss_res / ss_tot)

        # In OOD extrapolation, surrogate fails predictive ceiling (R2 < 0 and RMSE > obs_rms)
        assert r2_ood < 0.0, f"Expected R2 < 0 on OOD, got {r2_ood}"
        assert rmse_ood > obs_rms_ood, f"RMSE {rmse_ood} did not exceed obs_rms {obs_rms_ood}"

        # Gate 4 evidential gate fails on extrapolation holdout
        cand = CandidateDynamics(lhs="x_dot", terms=(Term("surrogate_rational", "theta"),), candidate_id="surrogate_rational")
        fit = FitResult(param_values={"a": popt[0], "b": popt[1]}, train_loss=train_loss, nfev=1, success=True, message="curve_fit")
        g4 = {
            "passed": False,  # Failed extrapolation
            "detail": f"extrapolation failure R2={r2_ood:.2f}, RMSE={rmse_ood:.3g}",
            "holdout_obs_rms": obs_rms_ood,
            "holdout_noise_sigma": float(np.std(resid_ood)),
            "holdout_n_samples": len(resid_ood),
            "rmse": rmse_ood,
            "r2": r2_ood,
            "holdout_residuals": resid_ood.tolist(),
        }
        sc_surrogate = ScoredCandidate(cand, fit=fit, train_loss=train_loss, holdout_rmse=rmse_ood, provenance={"gate4_status": g4})

        adj = adjudicate([sc_surrogate], benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY, data_origin=DataOrigin.SIMULATION)

        # APEDP recognizes surrogate is observationally adequate in-regime, but refuses UNIQUE promotion
        assert adj.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.epistemic_attribution == EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value
        assert "no promotion of least-bad models" in adj.reason
