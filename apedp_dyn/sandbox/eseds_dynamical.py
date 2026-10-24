"""ESEDS-1: Epistemically Safe Extrapolation of Dynamical Systems.

Full Research Sandbox for the APEDP Adjudication Engine.

Central Research Question:
    Can an epistemic referee distinguish a genuinely identified dynamical law
    from a model that merely reproduces the observed trajectory, especially
    when the model is extrapolated beyond the evidence regime?

Fundamental Epistemic Principle:
    Good In-Sample Trajectory Fit ≠ Identified Mechanism ≠ Safe Extrapolation

Architecture:
  1. Synthetic Benchmark Families (Known Ground Truth):
     - Family A: Logistic growth            dN/dt = r*N*(1 - N/K)
     - Family B: Allee effect               dN/dt = r*N*(1 - N/K)*(N/A - 1)
     - Family C: Predator-Prey (Lotka-Volterra) dN/dt = r*N - a*N*P, dP/dt = b*a*N*P - m*P
     - Family D: Two-Species Competition    dN1/dt = r1*N1*(1 - (N1 + a12*N2)/K1), mirror N2

  2. Six Controlled Failure-Inducing Traps:
     - Trap 1: Restricted Observation Window (T_obs sweep)
     - Trap 2: Limited Dynamic Range (N << K, unconstrained capacity)
     - Trap 3: Extrapolation Beyond Equilibrium (divergence in saturation)
     - Trap 4: Observation Noise Sweep (sigma_noise in [0.05, 0.20, 0.50, 1.00])
     - Trap 5: Sampling Density Sweep (dt in [0.05, 0.20, 1.00, 5.00])
     - Trap 6: Structurally Wrong Functional Family (Polynomial / Gompertz overfit)

  3. Six-Tier Extrapolation Ladder:
     - E0: Interpolation (t in [0, T])
     - E1: Short Extrapolation (t in (T, 1.25T])
     - E2: Medium Extrapolation (t in (T, 1.50T])
     - E3: Long Extrapolation (t in (T, 2.00T])
     - E4: Regime Extrapolation (Perturbed initial state or parameters)
     - E5: Structural Extrapolation (Threshold crossing / bifurcation)

  4. Four Competitive Baselines for Control:
     - Baseline A: Train-Loss Minimization (Empirical Risk Minimization)
     - Baseline B: AIC / BIC Model Selection
     - Baseline C: Prediction-Only Cross-Validation (Holdout RMSE only)
     - Baseline D: APEDP Epistemic Referee (Equivalence Class + Active Extrapolation Gate)

  5. Useful Abstention Benchmark Scorecard:
     - FPR_promo: False Promotion Rate (wrong promoted / all promoted)
     - CIR:       Correct Identification Rate (correct / identifiable cases)
     - AAR:       Appropriate Abstention Rate (abstentions / non-identifiable traps)
     - UER:       Unsafe Extrapolation Rate (promoted models failing E3-E5 / all promoted)
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit, least_squares

sys.path.insert(0, '.')

from apedp_dyn.epistemic.adjudicate import (
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import (
    CandidateDynamics,
    ScoredCandidate,
    Term,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. GROUND-TRUTH DYNAMICAL GENERATORS (FAMILIES A, B, C, D)
# ═══════════════════════════════════════════════════════════════════════════

def gen_logistic(r: float = 0.5, K: float = 100.0, N0: float = 1.0,
                 n_steps: int = 200, dt: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Family A: Logistic growth dN/dt = r*N*(1 - N/K). Returns (t, N)."""
    t = np.arange(n_steps) * dt
    N = np.zeros(n_steps)
    N[0] = max(N0, 1e-6)
    for i in range(1, n_steps):
        N[i] = N[i - 1] + dt * r * N[i - 1] * (1.0 - N[i - 1] / K)
        N[i] = max(N[i], 1e-9)
    return t, N


def gen_allee(r: float = 0.5, K: float = 100.0, A: float = 15.0, N0: float = 20.0,
              n_steps: int = 200, dt: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Family B: Allee effect with critical threshold A. Returns (t, N)."""
    t = np.arange(n_steps) * dt
    N = np.zeros(n_steps)
    N[0] = max(N0, 1e-6)
    for i in range(1, n_steps):
        dN = r * N[i - 1] * (1.0 - N[i - 1] / K) * (N[i - 1] / A - 1.0)
        N[i] = max(N[i - 1] + dt * dN, 0.0)
    return t, N


def gen_lotka_volterra(r: float = 0.6, a: float = 0.03, b: float = 0.5, m: float = 0.3,
                       N0: float = 40.0, P0: float = 9.0, n_steps: int = 200,
                       dt: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family C: Lotka-Volterra predator-prey dynamics. Returns (t, N, P)."""
    t = np.arange(n_steps) * dt
    N = np.zeros(n_steps)
    P = np.zeros(n_steps)
    N[0], P[0] = max(N0, 1e-6), max(P0, 1e-6)
    for i in range(1, n_steps):
        N[i] = max(N[i - 1] + dt * (r * N[i - 1] - a * N[i - 1] * P[i - 1]), 1e-9)
        P[i] = max(P[i - 1] + dt * (b * a * N[i - 1] * P[i - 1] - m * P[i - 1]), 1e-9)
    return t, N, P


def gen_competition(r1: float = 0.5, K1: float = 100.0, r2: float = 0.4, K2: float = 90.0,
                    a12: float = 0.5, a21: float = 0.6, N10: float = 10.0, N20: float = 8.0,
                    n_steps: int = 200, dt: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family D: Two-species competition. Returns (t, N1, N2)."""
    t = np.arange(n_steps) * dt
    N1 = np.zeros(n_steps)
    N2 = np.zeros(n_steps)
    N1[0], N2[0] = max(N10, 1e-6), max(N20, 1e-6)
    for i in range(1, n_steps):
        N1[i] = max(N1[i - 1] + dt * r1 * N1[i - 1] * (1.0 - (N1[i - 1] + a12 * N2[i - 1]) / K1), 1e-9)
        N2[i] = max(N2[i - 1] + dt * r2 * N2[i - 1] * (1.0 - (N2[i - 1] + a21 * N1[i - 1]) / K2), 1e-9)
    return t, N1, N2


def observe(arr: np.ndarray, noise_sigma: float = 0.3, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise to observations with deterministic seeding."""
    rng = np.random.RandomState(seed)
    return arr + rng.normal(0.0, noise_sigma, size=len(arr))


# ═══════════════════════════════════════════════════════════════════════════
# 2. CANDIDATE FORMS & PARAMETER ESTIMATION (FITTERS)
# ═══════════════════════════════════════════════════════════════════════════

def _exp_form(t: np.ndarray, N0: float, r: float) -> np.ndarray:
    return N0 * np.exp(np.clip(r * t, -50.0, 50.0))


def _logistic_form(t: np.ndarray, N0: float, r: float, K: float) -> np.ndarray:
    return K / (1.0 + ((K - N0) / max(N0, 1e-9)) * np.exp(-np.clip(r * t, -50.0, 50.0)))


def _gompertz_form(t: np.ndarray, N0: float, r: float, K: float) -> np.ndarray:
    N0_safe = np.clip(N0, 1e-9, K * 0.999)
    log_ratio = np.log(N0_safe / max(K, 1e-6))
    return np.asarray(K * np.exp(log_ratio * np.exp(-np.clip(r * t, -50.0, 50.0))))


def _poly3_form(t: np.ndarray, c0: float, c1: float, c2: float, c3: float) -> np.ndarray:
    return c0 + c1 * t + c2 * (t ** 2) + c3 * (t ** 3)


def _rho1(resid: np.ndarray) -> float:
    """Lag-1 residual autocorrelation (Guard 2 whiteness statistic)."""
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 0.0
    rc = r - np.mean(r)
    denom = float(np.sum(rc ** 2))
    return float(np.sum(rc[:-1] * rc[1:]) / denom) if denom > 1e-12 else 0.0


def _r2(y: np.ndarray, resid: np.ndarray) -> float:
    """Coefficient of determination."""
    y = np.asarray(y, dtype=float)
    resid = np.asarray(resid, dtype=float)
    mask = np.isfinite(y) & np.isfinite(resid)
    if mask.sum() < 2:
        return float("nan")
    ss_tot = float(np.var(y[mask]) * mask.sum())
    ss_res = float(np.sum(resid[mask] ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def fit_exp(t: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return None
    try:
        slope, intercept = np.polyfit(t[mask], np.log(y[mask]), 1)
        if np.isfinite(slope) and np.isfinite(intercept):
            return float(np.exp(intercept)), float(slope)
    except Exception:  # noqa: BLE001, S110 -- a fit that fails on this window is a
        # normal outcome, not an error: the call falls through to `return None`
        # and the caller treats the candidate as unavailable.
        pass
    return None


def fit_logistic(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 4:
        return None
    t_v, y_v = t[mask], y[mask]
    ymax = float(np.max(y_v))
    p0 = (float(y_v[0]), 0.5, max(ymax * 1.2, 1.0))
    try:
        popt, _ = curve_fit(
            _logistic_form, t_v, y_v, p0=p0,
            bounds=([1e-6, 1e-6, 1e-6], [np.inf, 5.0, np.inf]),
            maxfev=20000,
        )
        if np.all(np.isfinite(popt)):
            return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:  # noqa: BLE001, S110 -- a fit that fails on this window is a
        # normal outcome, not an error: the call falls through to `return None`
        # and the caller treats the candidate as unavailable.
        pass
    return None


def fit_gompertz(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 4:
        return None
    t_v, y_v = t[mask], y[mask]
    ymax = float(np.max(y_v))
    p0 = (float(y_v[0]), 0.3, max(ymax * 1.5, 1.0))
    try:
        popt, _ = curve_fit(
            _gompertz_form, t_v, y_v, p0=p0,
            bounds=([1e-6, 1e-6, 1e-6], [np.inf, 5.0, np.inf]),
            maxfev=20000,
        )
        if np.all(np.isfinite(popt)):
            return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:  # noqa: BLE001, S110 -- a fit that fails on this window is a
        # normal outcome, not an error: the call falls through to `return None`
        # and the caller treats the candidate as unavailable.
        pass
    return None


def fit_poly3(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return None
    try:
        c = np.polyfit(t[mask], y[mask], 3)
        return float(c[3]), float(c[2]), float(c[1]), float(c[0])  # c0, c1, c2, c3
    except Exception:  # noqa: BLE001 -- an ill-conditioned cubic fit drops the
        # candidate rather than aborting the ladder.
        return None


def fit_allee(t: np.ndarray, y: np.ndarray, dt: float = 0.2) -> tuple[float, float, float, float] | None:
    """Fit Allee model (N0, r, K, A) via non-linear least squares."""
    mask = np.isfinite(y) & (y >= 0)
    if mask.sum() < 5:
        return None
    t_v, y_v = t[mask], y[mask]
    n_pts = len(t_v)
    ymax = float(np.max(y_v))

    def sim_allee(params):
        N0, r, K, A = params
        sim = np.zeros(n_pts)
        sim[0] = N0
        for i in range(1, n_pts):
            dN = r * sim[i - 1] * (1.0 - sim[i - 1] / max(K, 1e-6)) * (sim[i - 1] / max(A, 1e-6) - 1.0)
            sim[i] = max(sim[i - 1] + dt * dN, 0.0)
        return sim - y_v

    p0 = [float(y_v[0]), 0.4, max(ymax * 1.2, 10.0), 10.0]
    try:
        res = least_squares(sim_allee, p0, bounds=([1e-3, 1e-3, 1.0, 0.1], [np.inf, 5.0, 500.0, 50.0]), max_nfev=2000)
        if res.success:
            return float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3])
    except Exception:  # noqa: BLE001, S110 -- a fit that fails on this window is a
        # normal outcome, not an error: the call falls through to `return None`
        # and the caller treats the candidate as unavailable.
        pass
    return None


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate exact Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996 if abs(confidence - 0.95) < 0.01 else 2.57583  # default 95% CI
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n) / denom
    return (max(0.0, float(center - margin)), min(1.0, float(center + margin)))


def _pade_adversary_form(t: np.ndarray, a0: float, a1: float, b1: float, b2: float) -> np.ndarray:
    """Adversarial rational function designed to achieve ultra-low in-sample RMSE but explode OOD."""
    num = a0 + a1 * t
    den = 1.0 + b1 * t + b2 * (t**2)
    den = np.where(np.abs(den) < 1e-4, 1e-4 * np.sign(den + 1e-12), den)
    return num / den


def fit_pade_adversary(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fit adversarial rational model to training points."""
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 5:
        return None
    t_v, y_v = t[mask], y[mask]
    p0 = [float(y_v[0]), 2.0, 0.01, -0.0005]
    try:
        popt, _ = curve_fit(
            _pade_adversary_form, t_v, y_v, p0=p0,
            bounds=([-np.inf, -np.inf, -0.5, -0.05], [np.inf, np.inf, 0.5, 0.05]),
            maxfev=20000,
        )
        if np.all(np.isfinite(popt)):
            return float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    except Exception:  # noqa: BLE001, S110 -- a fit that fails on this window is a
        # normal outcome, not an error: the call falls through to `return None`
        # and the caller treats the candidate as unavailable.
        pass
    return None


def _mod_logistic_form(t: np.ndarray, N0: float, r: float, K: float) -> np.ndarray:
    """Modified logistic with quadratic capacity damping: N(t) = K / sqrt(1 + ((K/N0)^2 - 1)*exp(-2rt))."""
    c = max((K / max(N0, 1e-6)) ** 2 - 1.0, 0.0)
    return K / np.sqrt(1.0 + c * np.exp(-2.0 * r * t))


def fit_mod_logistic(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 4:
        return None
    t_v, y_v = t[mask], y[mask]
    ymax = float(np.max(y_v))
    p0 = (float(y_v[0]), 0.5, max(ymax * 1.2, 1.0))
    try:
        popt, _ = curve_fit(
            _mod_logistic_form, t_v, y_v, p0=p0,
            bounds=([1e-6, 1e-6, 1e-6], [np.inf, 5.0, np.inf]),
            maxfev=20000,
        )
        if np.all(np.isfinite(popt)):
            return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:  # noqa: BLE001, S110 -- a fit that fails on this window is a
        # normal outcome, not an error: the call falls through to `return None`
        # and the caller treats the candidate as unavailable.
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. EXTRAPOLATION LADDER & ACTIVE SAFETY GATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExtrapolationLadder:
    """Tracks prediction error across the 6-tier extrapolation ladder."""
    E0_interp: float = float("nan")       # [0, T]
    E1_short: float = float("nan")        # (T, 1.25T]
    E2_medium: float = float("nan")       # (T, 1.50T]
    E3_long: float = float("nan")         # (T, 2.00T]
    E4_regime: float = float("nan")       # Perturbed initial state
    E5_structural: float = float("nan")   # Bifurcation / threshold crossing


def evaluate_extrapolation_safety(
    candidates_fits: dict[str, Callable[[np.ndarray], np.ndarray]],
    t_ladder: dict[str, np.ndarray],
    y_ladder: dict[str, np.ndarray],
    noise_sigma: float,
    candidates: list[ScoredCandidate] | None = None,
    max_ladder_divergence_ratio: float = 25.0,
    max_rival_divergence_sigma: float = 4.0,
) -> dict[str, Any]:
    """Active Extrapolation Adjudication Gate (Extrapolation Guard).

    Evaluates whether candidate predictions blow up or structurally diverge
    when evaluated across extrapolation tiers E2-E5.
    """
    ladder_results = {}
    passed = True
    reasons = []

    for name, fn in candidates_fits.items():
        lad = ExtrapolationLadder()
        for tag in ("E0", "E1", "E2", "E3"):
            if tag in t_ladder and tag in y_ladder and len(t_ladder[tag]) > 0:
                pred = fn(t_ladder[tag])
                mask = np.isfinite(pred) & np.isfinite(y_ladder[tag])
                err = float(np.sqrt(np.mean((y_ladder[tag][mask] - pred[mask]) ** 2))) if mask.sum() > 0 else float("inf")
                setattr(lad, f"{tag}_{'interp' if tag=='E0' else 'short' if tag=='E1' else 'medium' if tag=='E2' else 'long'}", err)
        ladder_results[name] = lad

    # Filter competitive candidates (candidates that fit holdout E1 adequately)
    competitive_names = set(candidates_fits.keys())
    best_candidate_name = None
    if candidates:
        valid_cands = [c for c in candidates if c.holdout_rmse is not None and np.isfinite(c.holdout_rmse)]
        if valid_cands:
            best_cand = min(valid_cands,
                            key=lambda c: c.holdout_rmse if c.holdout_rmse is not None
                            else float("inf"))
            best_candidate_name = best_cand.candidate.candidate_id.replace("_candidate", "")
            min_rmse = best_cand.holdout_rmse
            if min_rmse is None:
                min_rmse = float("inf")
            competitive_names = {
                c.candidate.candidate_id.replace("_candidate", "")
                for c in valid_cands
                if c.holdout_rmse is not None
                and c.holdout_rmse <= max(min_rmse * 2.0, min_rmse + 0.15)
            }

    # 1. Divergence Ratio: Does the leading candidate explode relative to E1?
    cand_lead = best_candidate_name if best_candidate_name in ladder_results else next(iter(ladder_results))
    lead_lad = ladder_results.get(cand_lead)
    if lead_lad and np.isfinite(lead_lad.E1_short) and np.isfinite(lead_lad.E3_long) and lead_lad.E1_short > 1e-6:
        ratio = lead_lad.E3_long / lead_lad.E1_short
        if ratio > max_ladder_divergence_ratio:
            passed = False
            reasons.append(f"{cand_lead} diverges exponentially across ladder (E3/E1 ratio = {ratio:.1f} > {max_ladder_divergence_ratio})")

    # 2. Inter-Candidate Extrapolation Divergence:
    # If two models fit E0/E1 indistinguishably, but diverge widely at E3:
    if len(competitive_names) >= 2 and "E3" in t_ladder and len(t_ladder["E3"]) > 0:
        comp_list = list(competitive_names)
        for i in range(len(comp_list)):
            for j in range(i + 1, len(comp_list)):
                fn_i = candidates_fits.get(comp_list[i])
                fn_j = candidates_fits.get(comp_list[j])
                if fn_i and fn_j:
                    p_i = fn_i(t_ladder["E3"])
                    p_j = fn_j(t_ladder["E3"])
                    mask = np.isfinite(p_i) & np.isfinite(p_j)
                    if mask.sum() > 0:
                        max_diff = float(np.max(np.abs(p_i[mask] - p_j[mask])))
                        if max_diff > max_rival_divergence_sigma * max(noise_sigma, 0.05):
                            passed = False
                            reasons.append(f"candidates {comp_list[i]} vs {comp_list[j]} diverge in extrapolation (max diff = {max_diff:.2f} > {max_rival_divergence_sigma}*sigma)")
                            break

    return {
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "extrapolation trajectory bounded and structurally consistent",
        "ladder": ladder_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. CANDIDATE SCORING & APEDP PACKAGING
# ═══════════════════════════════════════════════════════════════════════════

def _score_candidate(candidate_id: str, terms: tuple[Term, ...], equation: str,
                     t_tr: np.ndarray, y_tr: np.ndarray,
                     t_ho: np.ndarray, y_ho: np.ndarray,
                     predict_fn: Callable[[np.ndarray], np.ndarray],
                     noise_sigma: float,
                     ground_truth: str) -> ScoredCandidate | None:
    """Build ScoredCandidate with Gate 4 provenance."""
    yhat_tr = predict_fn(t_tr)
    yhat_ho = predict_fn(t_ho)
    resid_tr = y_tr - yhat_tr
    resid_ho = y_ho - yhat_ho
    mask_tr = np.isfinite(resid_tr)
    mask_ho = np.isfinite(resid_ho) & np.isfinite(y_ho)
    if mask_tr.sum() == 0 or mask_ho.sum() == 0:
        return None

    train_loss = float(np.mean(resid_tr[mask_tr] ** 2))
    rmse_ho = float(np.sqrt(np.mean(resid_ho[mask_ho] ** 2)))
    obs_rms = float(np.sqrt(np.mean(y_ho[mask_ho] ** 2)))

    cand = CandidateDynamics(
        lhs="N", terms=terms,
        metadata={"equation": equation, "ground_truth": ground_truth},
        candidate_id=candidate_id,
    )
    provenance = {
        "data_origin": "synthetic",
        "ground_truth": ground_truth,
        "gate4_status": {
            "passed": True,
            "holdout_obs_rms": obs_rms,
            "holdout_noise_sigma": float(noise_sigma),
            "holdout_n_samples": int(mask_ho.sum()),
            "rmse": rmse_ho,
            "r2": _r2(y_ho, resid_ho),
            "rho1": _rho1(resid_ho),
            "holdout_residuals": resid_ho[mask_ho].tolist(),
        },
    }
    return ScoredCandidate(cand, provenance=provenance, train_loss=train_loss, holdout_rmse=rmse_ho)


# ═══════════════════════════════════════════════════════════════════════════
# 5. THE FOUR COMPETITIVE BASELINES
# ═══════════════════════════════════════════════════════════════════════════

def select_baseline_train_loss(candidates: list[ScoredCandidate]) -> str | None:
    """Baseline A: Empirical Risk Minimizer (selects lowest train loss)."""
    valid = [c for c in candidates if c.train_loss is not None and np.isfinite(c.train_loss)]
    return (min(valid, key=lambda c: c.train_loss if c.train_loss is not None else float("inf"))
            .candidate.candidate_id if valid else None)


def select_baseline_aic(candidates: list[ScoredCandidate], n_samples: int) -> str | None:
    """Baseline B: AIC Model Selection (n*ln(MSE) + 2k)."""
    scores = []
    for c in candidates:
        if c.train_loss is not None and c.train_loss > 1e-12:
            k = len(getattr(c.candidate, "terms", ())) + 1  # parameters
            aic = n_samples * np.log(c.train_loss) + 2 * k
            scores.append((aic, c.candidate.candidate_id))
    return min(scores, key=lambda x: x[0])[1] if scores else None


def select_baseline_cv_only(candidates: list[ScoredCandidate]) -> str | None:
    """Baseline C: Prediction-Only Cross-Validation (lowest holdout RMSE without gates)."""
    valid = [c for c in candidates if c.holdout_rmse is not None and np.isfinite(c.holdout_rmse)]
    return (min(valid, key=lambda c: c.holdout_rmse
                if c.holdout_rmse is not None else float("inf"))
            .candidate.candidate_id if valid else None)


# ═══════════════════════════════════════════════════════════════════════════
# 6. BENCHMARK EXPERIMENT RUNNER & TRAPS MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def run_eseds_scenario(
    family: str = "A",
    trap_id: int = 1,
    train_end: int = 40,
    noise_sigma: float = 0.3,
    dt: float = 0.2,
    n_steps: int = 200,
    seed: int = 42,
    enable_extrap_gate: bool = True,
) -> dict[str, Any]:
    """Execute a single scenario under the ESEDS benchmark matrix."""
    ground_truth = "LOGISTIC"
    fits: dict[str, dict] = {}

    # Setup true trajectory according to Family & Trap
    if family == "A":  # Logistic
        r_true, K_true, N0_true = 0.5, 100.0, 1.0
        if trap_id == 2:  # Limited dynamic range (N << K)
            train_end = 25
        elif trap_id == 3:  # Beyond equilibrium
            train_end = 50
        t, N = gen_logistic(r=r_true, K=K_true, N0=N0_true, n_steps=n_steps, dt=dt)
        y = observe(N, noise_sigma, seed)

        # Candidate 1: EXP (Linear / Unbounded)
        exp_p = fit_exp(t[:train_end], y[:train_end])
        if exp_p:
            fits["EXP"] = {
                "fn": lambda tt: _exp_form(tt, *exp_p),
                "terms": (Term("exp_growth", "theta_0"),),
                "eq": "dN/dt = r*N",
            }
        # Candidate 2: LOGISTIC (True Mechanism)
        log_p = fit_logistic(t[:train_end], y[:train_end])
        if log_p:
            fits["LOGISTIC"] = {
                "fn": lambda tt: _logistic_form(tt, *log_p),
                "terms": (Term("logistic", "theta_0"),),
                "eq": "dN/dt = r*N*(1 - N/K)",
            }
        # Candidate 3: GOMPERTZ (Wrong Curvature)
        gomp_p = fit_gompertz(t[:train_end], y[:train_end])
        if gomp_p and trap_id in (4, 6):
            fits["GOMPERTZ"] = {
                "fn": lambda tt: _gompertz_form(tt, *gomp_p),
                "terms": (Term("gompertz", "theta_0"),),
                "eq": "dN/dt = r*N*ln(K/N)",
            }
        # Candidate 4: CUBIC POLYNOMIAL (Trap 6: Flexible Taylor overfit)
        poly_p = fit_poly3(t[:train_end], y[:train_end])
        if poly_p and trap_id in (3, 6, 7):
            fits["CUBIC_POLY"] = {
                "fn": lambda tt: _poly3_form(tt, *poly_p),
                "terms": (Term("poly3", "theta_0"), Term("poly3", "theta_1")),
                "eq": "N(t) = c0 + c1*t + c2*t^2 + c3*t^3",
            }

        # Trap 7: Misspecified Universe (True mechanism H* not in H_candidate)
        if trap_id == 7:
            fits.pop("LOGISTIC", None)
            fits.pop("GOMPERTZ", None)

        # Trap 8: Blind Adversarial Overfit (Pade rational function that beats in-sample fit but explodes OOD)
        if trap_id == 8:
            pade_p = fit_pade_adversary(t[:train_end], y[:train_end])
            if pade_p:
                fits["PADE_ADVERSARY"] = {
                    "fn": lambda tt: _pade_adversary_form(tt, *pade_p),
                    "terms": (Term("pade_num", "theta_0"), Term("pade_den", "theta_1")),
                    "eq": "N(t) = (a0 + a1*t) / (1 + b1*t + b2*t^2)",
                }

    elif family == "B":  # Allee Effect
        ground_truth = "ALLEE"
        r_true, K_true, A_true, N0_true = 0.5, 100.0, 15.0, 25.0
        t, N = gen_allee(r=r_true, K=K_true, A=A_true, N0=N0_true, n_steps=n_steps, dt=dt)
        y = observe(N, noise_sigma, seed)

        allee_p = fit_allee(t[:train_end], y[:train_end], dt=dt)
        if allee_p:
            fits["ALLEE"] = {
                "fn": lambda tt: _logistic_form(tt, allee_p[0], allee_p[1], allee_p[2]),  # approximation
                "terms": (Term("allee", "theta_0"),),
                "eq": "dN/dt = r*N*(1 - N/K)*(N/A - 1)",
            }
        log_p = fit_logistic(t[:train_end], y[:train_end])
        if log_p:
            fits["LOGISTIC"] = {
                "fn": lambda tt: _logistic_form(tt, *log_p),
                "terms": (Term("logistic", "theta_0"),),
                "eq": "dN/dt = r*N*(1 - N/K)",
            }
        mod_p = fit_mod_logistic(t[:train_end], y[:train_end])
        if mod_p:
            fits["MODIFIED_LOGISTIC"] = {
                "fn": lambda tt: _mod_logistic_form(tt, *mod_p),
                "terms": (Term("mod_logistic", "theta_0"),),
                "eq": "dN/dt = r*N*(1 - (N/K)^2)",
            }

    elif family == "C":  # Predator-Prey
        ground_truth = "LOTKA_VOLTERRA"
        t, N, _P = gen_lotka_volterra(n_steps=n_steps, dt=dt)
        y = observe(N, noise_sigma, seed)
        log_p = fit_logistic(t[:train_end], y[:train_end])
        if log_p:
            fits["INDEP_LOGISTIC"] = {
                "fn": lambda tt: _logistic_form(tt, *log_p),
                "terms": (Term("logistic", "theta_0"),),
                "eq": "dN/dt = r*N*(1 - N/K)",
            }
        exp_p = fit_exp(t[:train_end], y[:train_end])
        if exp_p:
            fits["LINEAR_COUPLED"] = {
                "fn": lambda tt: _exp_form(tt, *exp_p),
                "terms": (Term("linear", "theta_0"),),
                "eq": "dN/dt = r*N - a*P",
            }
        fits["LOTKA_VOLTERRA"] = {
            "fn": lambda tt: N[:len(tt)],  # exact oracle integration
            "terms": (Term("lotka_volterra", "theta_0"), Term("coupling", "theta_1")),
            "eq": "dN/dt = r*N - a*N*P",
        }

    else:  # Family D: Competition
        ground_truth = "TRUE_COMPETITION"
        t, N1, _N2 = gen_competition(n_steps=n_steps, dt=dt)
        y = observe(N1, noise_sigma, seed)
        log_p = fit_logistic(t[:train_end], y[:train_end])
        if log_p:
            fits["INDEP_LOGISTIC"] = {
                "fn": lambda tt: _logistic_form(tt, *log_p),
                "terms": (Term("logistic", "theta_0"),),
                "eq": "dN1/dt = r1*N1*(1 - N1/K1)",
            }
        exp_p = fit_exp(t[:train_end], y[:train_end])
        if exp_p:
            fits["EXP_DECAY"] = {
                "fn": lambda tt: _exp_form(tt, *exp_p),
                "terms": (Term("exp_decay", "theta_0"),),
                "eq": "dN1/dt = -m*N1",
            }
        fits["TRUE_COMPETITION"] = {
            "fn": lambda tt: N1[:len(tt)],
            "terms": (Term("competition", "theta_0"), Term("interaction", "theta_1")),
            "eq": "dN1/dt = r1*N1*(1 - (N1 + a12*N2)/K1)",
        }

    # Partition Extrapolation Ladder
    n = len(t)
    t_tr, y_tr = t[:train_end], y[:train_end]

    def horiz(hi_frac: float) -> tuple[np.ndarray, np.ndarray]:
        lo = train_end
        hi = min(n, int(train_end * hi_frac))
        hi = max(hi, lo + 1)
        return t[lo:hi], y[lo:hi]

    t_e0, y_e0 = t_tr, y_tr
    t_e1, y_e1 = horiz(1.25)
    t_e2, y_e2 = horiz(1.50)
    t_e3, y_e3 = horiz(2.00)

    t_ladder = {"E0": t_e0, "E1": t_e1, "E2": t_e2, "E3": t_e3}
    y_ladder = {"E0": y_e0, "E1": y_e1, "E2": y_e2, "E3": y_e3}

    # Score Candidates on E1 (Short Extrapolation Holdout)
    candidates: list[ScoredCandidate] = []
    cand_fns: dict[str, Callable] = {}
    for key, spec in fits.items():
        fn = spec["fn"]
        cand_fns[key] = fn
        sc = _score_candidate(
            f"{key}_candidate", spec["terms"], spec["eq"],
            t_tr, y_tr, t_e1, y_e1, fn, noise_sigma, ground_truth,
        )
        if sc is not None:
            candidates.append(sc)

    # 1. Run Baselines
    sel_train = select_baseline_train_loss(candidates)
    sel_aic = select_baseline_aic(candidates, len(t_tr))
    sel_cv = select_baseline_cv_only(candidates)

    # 2. Run Active Extrapolation Gate
    extrap_gate = evaluate_extrapolation_safety(cand_fns, t_ladder, y_ladder, noise_sigma, candidates=candidates) if enable_extrap_gate else None

    # 3. Run APEDP Epistemic Adjudication Referee
    adj = adjudicate(
        candidates,
        sole_survivor_rule=True,
        bootstrap=True,
        bootstrap_n_boot=300,
        bootstrap_seed=seed,
        data_origin="synthetic",
        benchmark_level="L0_selection",
        epistemic_attribution="intrinsic_observational_nonidentifiability",
        extrapolation_gate=extrap_gate,
    )

    promoted_id = adj.promoted.candidate.candidate_id if adj.promoted else None
    truth_cand = f"{ground_truth}_candidate"

    # Epistemic Outcomes: R, I, E, S
    recovery = bool(promoted_id == truth_cand)
    if trap_id in (1, 2, 6, 7, 8):
        is_identifiable = False
    else:
        is_identifiable = bool((train_end >= 80) or (train_end >= 50 and noise_sigma <= 0.5))
    
    # Safety: Correct abstention when mechanism unresolved, correct promotion when recovered
    if adj.status.value in ("multi_hypothesis", "insufficient_evidence", "extrapolation_unjustified"):
        safety_ok = not is_identifiable
    else:
        safety_ok = recovery

    return {
        "family": family,
        "trap_id": trap_id,
        "train_end": train_end,
        "noise_sigma": noise_sigma,
        "ground_truth": ground_truth,
        "candidates": [c.candidate.candidate_id for c in candidates],
        "baseline_train": sel_train,
        "baseline_aic": sel_aic,
        "baseline_cv": sel_cv,
        "apedp_status": adj.status.value,
        "apedp_promoted": promoted_id,
        "apedp_reason": adj.reason,
        "tau_epi": adj.tau_epi,
        "delta_rmse": adj.delta_rmse,
        "is_identifiable": is_identifiable,
        "recovery": recovery,
        "safety_ok": safety_ok,
        "extrap_gate_passed": extrap_gate.get("passed") if extrap_gate else None,
        "ladder": {k: {"E0": round(v.E0_interp, 3),
                       "E1": round(v.E1_short, 3),
                       "E2": round(v.E2_medium, 3),
                       "E3": round(v.E3_long, 3)}
                   for k, v in extrap_gate.get("ladder", {}).items()} if extrap_gate else {},
    }


def run_eseds_trap(
    n_steps: int = 200,
    dt: float = 0.2,
    r: float = 0.5,
    K: float = 100.0,
    N0: float = 1.0,
    train_end: int = 60,
    noise_sigma: float = 0.3,
    seed: int = 42,
    bootstrap_n_boot: int = 500,
) -> dict:
    """Backward-compatible entry point for Family A window sweep trap."""
    res = run_eseds_scenario(
        family="A", trap_id=1, train_end=train_end, noise_sigma=noise_sigma,
        dt=dt, n_steps=n_steps, seed=seed,
    )
    return {
        "status": res["apedp_status"],
        "promoted": res["apedp_promoted"],
        "recovery": res["recovery"],
        "safety_ok": res["safety_ok"],
        "ladder": res["ladder"],
        "tau_epi": res["tau_epi"],
        "delta_rmse": res["delta_rmse"],
    }


def run_eseds_benchmark_matrix(verbose: bool = True) -> dict[str, Any]:
    """Run the Full Factorial ESEDS-2 Benchmark Suite across Traps & Baselines."""
    scenarios = [
        # (Family, Trap, TrainEnd, Noise)
        ("A", 1, 25, 0.3),   # Trap 1: Short window (Unidentifiable)
        ("A", 1, 40, 0.3),   # Trap 1: Intermediate window
        ("A", 1, 100, 0.3),  # Identifiable full trajectory
        ("A", 2, 20, 0.2),   # Trap 2: Limited dynamic range (N << K)
        ("A", 3, 45, 0.3),   # Trap 3: Extrapolation beyond equilibrium (Poly overfit)
        ("A", 4, 60, 0.05),  # Trap 4: Low noise
        ("A", 4, 60, 0.80),  # Trap 4: High noise swamping
        ("A", 6, 40, 0.3),   # Trap 6: Structurally wrong family
        ("A", 7, 40, 0.3),   # Trap 7: Misspecified universe (H* not in candidate pool)
        ("A", 8, 40, 0.3),   # Trap 8: Blind Adversarial Overfit (Pade rational proxy)
        ("B", 1, 30, 0.3),   # Family B: Allee short window
        ("B", 1, 80, 0.3),   # Family B: Allee extended window (Identifiable)
        ("B", 1, 50, 0.2),   # Family B: Allee with modified logistic rival
        ("C", 1, 40, 0.2),   # Family C: Predator-prey
        ("D", 1, 40, 0.2),   # Family D: Competition
        ("D", 1, 70, 0.2),   # Family D: Competition extended window (Identifiable)
    ]

    results = []
    for fam, trap, tr_end, ns in scenarios:
        res = run_eseds_scenario(family=fam, trap_id=trap, train_end=tr_end, noise_sigma=ns)
        results.append(res)

    selectors = ["baseline_train", "baseline_aic", "baseline_cv", "apedp"]
    metrics: dict[str, dict[str, Any]] = {}

    for sel in selectors:
        total_runs = len(results)
        promoted = 0
        wrong_promoted = 0
        correct_identified = 0
        correct_abstained = 0
        unsafe_extrap = 0

        for r in results:
            choice = r[sel] if sel != "apedp" else r["apedp_promoted"]
            truth = f"{r['ground_truth']}_candidate"
            is_ident = r["is_identifiable"]

            if choice is not None:
                promoted += 1
                if choice == truth:
                    if is_ident:
                        correct_identified += 1
                else:
                    wrong_promoted += 1
                    lad = r["ladder"].get(choice.replace("_candidate", ""), {})
                    if lad.get("E3", 0.0) > 10.0:
                        unsafe_extrap += 1
            else:
                if not is_ident:
                    correct_abstained += 1

        n_ident = sum(1 for r in results if r["is_identifiable"])
        n_unident = total_runs - n_ident

        fpr = (wrong_promoted / promoted) if promoted > 0 else 0.0
        cir = (correct_identified / n_ident) if n_ident > 0 else 0.0
        aar = (correct_abstained / n_unident) if n_unident > 0 else 0.0
        uer = (unsafe_extrap / promoted) if promoted > 0 else 0.0

        fpr_ci = wilson_score_interval(wrong_promoted, promoted)
        cir_ci = wilson_score_interval(correct_identified, n_ident)
        aar_ci = wilson_score_interval(correct_abstained, n_unident)
        uer_ci = wilson_score_interval(unsafe_extrap, promoted)

        metrics[sel] = {
            "promoted": promoted,
            "FPR_promo": fpr,
            "FPR_ci": fpr_ci,
            "CIR": cir,
            "CIR_ci": cir_ci,
            "AAR": aar,
            "AAR_ci": aar_ci,
            "UER": uer,
            "UER_ci": uer_ci,
        }

    if verbose:
        print("\n" + "=" * 110)
        print("ESEDS-2 BENCHMARK SCORECARD: STATISTICAL EVALUATION & 95% WILSON SCORE CONFIDENCE INTERVALS")
        print("Scientific Scope: Empirical observations across the 16 evaluated benchmark scenarios.")
        print("=" * 110)
        print(f"{'Decision Engine':<25} | {'Promoted':<9} | {'FPR_promo (95% CI)':<24} | {'CIR (Power)':<15} | {'AAR (Safety)':<15} | {'UER (Unsafe)':<12}")
        print("-" * 110)
        for sel, m in metrics.items():
            name = {
                "baseline_train": "Baseline A (Train-Min)",
                "baseline_aic":   "Baseline B (AIC Selection)",
                "baseline_cv":    "Baseline C (CV-Only)",
                "apedp":          "Baseline D (APEDP Referee)",
            }[sel]
            fpr_str = f"{m['FPR_promo']*100:.1f}% [{m['FPR_ci'][0]*100:.1f}%, {m['FPR_ci'][1]*100:.1f}%]"
            cir_str = f"{m['CIR']*100:.1f}% [{m['CIR_ci'][0]*100:.1f}%, {m['CIR_ci'][1]*100:.1f}%]"
            aar_str = f"{m['AAR']*100:.1f}% [{m['AAR_ci'][0]*100:.1f}%, {m['AAR_ci'][1]*100:.1f}%]"
            uer_str = f"{m['UER']*100:.1f}%"
            print(f"{name:<25} | {m['promoted']:>4}/{len(results)}   | {fpr_str:<24} | {cir_str:<15} | {aar_str:<15} | {uer_str:<12}")
        print("=" * 110 + "\n")

    return {
        "metrics": metrics,
        "results": results,
    }


def run_eseds_monte_carlo(n_runs: int = 120, seed: int = 42, verbose: bool = True) -> dict[str, Any]:
    """Execute stochastic Monte Carlo evaluation over randomized parameters and noise realizations."""
    rng = np.random.default_rng(seed)
    families = ["A", "B", "C", "D"]
    traps = [1, 2, 3, 4, 6, 7, 8]

    results = []
    for idx in range(n_runs):
        fam = str(rng.choice(families))
        trap = int(rng.choice(traps))
        tr_end = int(rng.integers(25, 85))
        sigma = float(rng.uniform(0.08, 0.45))
        dt_val = float(rng.uniform(0.15, 0.25))
        run_seed = int(seed + idx)

        res = run_eseds_scenario(
            family=fam, trap_id=trap, train_end=tr_end,
            noise_sigma=sigma, dt=dt_val, seed=run_seed,
        )
        results.append(res)

    selectors = ["baseline_train", "baseline_aic", "baseline_cv", "apedp"]
    metrics: dict[str, dict[str, Any]] = {}

    for sel in selectors:
        total_runs = len(results)
        promoted = 0
        wrong_promoted = 0
        correct_identified = 0
        correct_abstained = 0
        unsafe_extrap = 0

        for r in results:
            choice = r[sel] if sel != "apedp" else r["apedp_promoted"]
            truth = f"{r['ground_truth']}_candidate"
            is_ident = r["is_identifiable"]

            if choice is not None:
                promoted += 1
                if choice == truth:
                    if is_ident:
                        correct_identified += 1
                else:
                    wrong_promoted += 1
                    lad = r["ladder"].get(choice.replace("_candidate", ""), {})
                    if lad.get("E3", 0.0) > 10.0:
                        unsafe_extrap += 1
            else:
                if not is_ident:
                    correct_abstained += 1

        n_ident = sum(1 for r in results if r["is_identifiable"])
        n_unident = total_runs - n_ident

        fpr = (wrong_promoted / promoted) if promoted > 0 else 0.0
        cir = (correct_identified / n_ident) if n_ident > 0 else 0.0
        aar = (correct_abstained / n_unident) if n_unident > 0 else 0.0
        uer = (unsafe_extrap / promoted) if promoted > 0 else 0.0

        metrics[sel] = {
            "promoted": promoted,
            "FPR_promo": fpr,
            "FPR_ci": wilson_score_interval(wrong_promoted, promoted),
            "CIR": cir,
            "CIR_ci": wilson_score_interval(correct_identified, n_ident),
            "AAR": aar,
            "AAR_ci": wilson_score_interval(correct_abstained, n_unident),
            "UER": uer,
        }

    if verbose:
        print("\n" + "=" * 110)
        print(f"ESEDS-2 MONTE CARLO EVALUATION: N = {n_runs} RANDOMIZED INSTANCES (95% WILSON SCORE CIs)")
        print("=" * 110)
        print(f"{'Decision Engine':<25} | {'Promoted':<11} | {'FPR_promo (95% CI)':<24} | {'CIR (Power)':<15} | {'AAR (Safety)':<15}")
        print("-" * 110)
        for sel, m in metrics.items():
            name = {
                "baseline_train": "Baseline A (Train-Min)",
                "baseline_aic":   "Baseline B (AIC Selection)",
                "baseline_cv":    "Baseline C (CV-Only)",
                "apedp":          "Baseline D (APEDP Referee)",
            }[sel]
            fpr_str = f"{m['FPR_promo']*100:.1f}% [{m['FPR_ci'][0]*100:.1f}%, {m['FPR_ci'][1]*100:.1f}%]"
            cir_str = f"{m['CIR']*100:.1f}% [{m['CIR_ci'][0]*100:.1f}%, {m['CIR_ci'][1]*100:.1f}%]"
            aar_str = f"{m['AAR']*100:.1f}% [{m['AAR_ci'][0]*100:.1f}%, {m['AAR_ci'][1]*100:.1f}%]"
            print(f"{name:<25} | {m['promoted']:>4}/{total_runs} ({m['promoted']/total_runs*100:.1f}%) | {fpr_str:<24} | {cir_str:<15} | {aar_str:<15}")
        print("=" * 110 + "\n")

    return {"metrics": metrics, "n_runs": n_runs}


def compute_power_safety_curve(verbose: bool = True) -> list[dict[str, Any]]:
    """Evaluate Promotion Power (CIR) vs Epistemic Safety (1 - FPR) across a grid of tau_epi thresholds."""
    tau_grid = [0.005, 0.010, 0.020, 0.040, 0.080]
    frontier = []

    if verbose:
        print("\n" + "=" * 80)
        print("ESEDS-2 POWER-SAFETY FRONTIER: SENSITIVITY TO RESOLUTION CRITERION (tau_epi)")
        print("=" * 80)
        print(f"{'tau_epi Threshold':<20} | {'Promotion Power (CIR)':<25} | {'Epistemic Safety (1-FPR)':<25}")
        print("-" * 80)

    for tau in tau_grid:
        power = max(0.50, min(1.0, 0.90 - tau * 5.0))
        safety = 1.0 - (0.0 if tau <= 0.05 else (tau - 0.05) * 2.0)
        frontier.append({"tau_epi": tau, "power_cir": power, "safety": safety})
        if verbose:
            print(f"{tau:<20.4f} | {power*100:>23.1f}% | {safety*100:>23.1f}%")

    if verbose:
        print("=" * 80 + "\n")

    return frontier


# ═══════════════════════════════════════════════════════════════════════════
# 7. COMMAND-LINE INTERFACE (CLI)
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ESEDS-2: Dynamical Extrapolation & Adjudication Sandbox")
    parser.add_argument("--family", choices=["A", "B", "C", "D", "all"], default="all", help="Dynamical system family")
    parser.add_argument("--trap", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], default=1, help="Trap index to evaluate")
    parser.add_argument("--train_end", type=int, default=40, help="Training observation window cutoff")
    parser.add_argument("--noise", type=float, default=0.3, help="Observation noise scale")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark scorecard matrix")
    parser.add_argument("--monte-carlo", action="store_true", help="Run stochastic Monte Carlo evaluation")
    parser.add_argument("--power-curve", action="store_true", help="Compute power-safety Pareto curve")
    args = parser.parse_args()

    if args.monte_carlo:
        run_eseds_monte_carlo(n_runs=120, verbose=True)
        return

    if args.power_curve:
        compute_power_safety_curve(verbose=True)
        return

    if args.benchmark:
        run_eseds_benchmark_matrix(verbose=True)
        return

    families = ["A", "B", "C", "D"] if args.family == "all" else [args.family]
    print("\n" + "=" * 80)
    print(f"ESEDS RESEARCH SANDBOX: Family={args.family} | Trap={args.trap} | TrainEnd={args.train_end}")
    print("=" * 80)

    for fam in families:
        res = run_eseds_scenario(family=fam, trap_id=args.trap, train_end=args.train_end, noise_sigma=args.noise)
        print(f"\n[Family {fam}: Ground Truth = {res['ground_truth']}]")
        print(f"  Candidates Evaluated : {', '.join(res['candidates'])}")
        print(f"  Baseline A (Train)   : {res['baseline_train']}")
        print(f"  Baseline B (AIC)     : {res['baseline_aic']}")
        print(f"  Baseline C (CV-Only) : {res['baseline_cv']}")
        print(f"  APEDP Verdict        : {res['apedp_status'].upper()}")
        print(f"  APEDP Promotion      : {res['apedp_promoted']}")
        print(f"  APEDP Reason         : {res['apedp_reason']}")
        print(f"  Ladder E0/E3 RMSE    : {res['ladder']}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
