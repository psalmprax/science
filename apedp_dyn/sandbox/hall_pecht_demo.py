"""APEDP Hall-Petch grain-size strengthening demo.

Demonstrates the APEDP epistemic adjudication pipeline on the
Hall-Petch effect: sigma_y = sigma_0 + k·d^(-1/2).

Shows how APEDP's epistemic gates (gamma ceiling, whiteness,
bootstrap tiebreaker, data_origin labeling) work on a real
materials science problem using actual fitted regression models
(no random losses).
"""
from __future__ import annotations

import sys

sys.path.insert(0, '.')  # ensure apedp_dyn is importable


import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

# Frozen live-tree constants (from adjudicate.py history)
SOLE_SURVIVOR_GAMMA = 3.0
CAL_SIGMA = 1.8     # 36-cell grid median
RESID_RHO_FLAG = 0.5


# ── Hall-Petch data generation ──────────────────────────────────────────
def generate_hall_pecht_data(n_points: int = 50,
                             noise_sigma: float = 0.03,
                             sigma_0: float = 200.0,
                             k_hp: float = 50.0,
                             d_range: tuple[float, float] = (1e-6, 1e-3)):
    """Generate Hall-Petch synthetic data: sigma_y = sigma_0 + k·d^(-1/2) + noise."""
    d = np.logspace(-6, -3, n_points)  # grain size: 1μm to 1mm
    sigma_y_true = sigma_0 + k_hp * d**(-0.5)
    noise = np.random.RandomState(42).normal(0, noise_sigma * sigma_0, n_points)
    sigma_y = sigma_y_true + noise
    return d, sigma_y


# ── Fitted Candidate Generation ──────────────────────────────────────────
def fit_hall_pecht_candidates(d: np.ndarray,
                              sigma_y: np.ndarray,
                              n_train: int = 20,
                              noise_sigma: float = 0.03) -> list[ScoredCandidate]:
    """Fit actual structural hypotheses on training split and score on holdout split."""
    d_tr, y_tr = d[:n_train], sigma_y[:n_train]
    d_ho, y_ho = d[n_train:], sigma_y[n_train:]
    obs_rms = float(np.sqrt(np.mean(y_ho**2)))
    noise_est = float(noise_sigma * 200.0)

    candidates: list[ScoredCandidate] = []

    # 1. True Hall-Petch model: sigma_y = a0 + a1 * d^(-1/2)
    X_tr_hp = np.column_stack([np.ones_like(d_tr), d_tr**(-0.5)])
    w_hp, *_ = np.linalg.lstsq(X_tr_hp, y_tr, rcond=None)
    pred_tr_hp = X_tr_hp @ w_hp
    loss_tr_hp = float(np.mean((y_tr - pred_tr_hp)**2))
    X_ho_hp = np.column_stack([np.ones_like(d_ho), d_ho**(-0.5)])
    pred_ho_hp = X_ho_hp @ w_hp
    resid_hp = y_ho - pred_ho_hp
    rmse_ho_hp = float(np.sqrt(np.mean(resid_hp**2)))

    c_hp = CandidateDynamics(
        lhs="yield_strength",
        terms=(Term("one", "a0"), Term("d_inv_half", "a1")),
        candidate_id="hp_true",
    )
    candidates.append(ScoredCandidate(
        c_hp,
        provenance={
            "data_origin": "simulation",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(d_ho),
                "rmse": rmse_ho_hp,
                "holdout_residuals": resid_hp.tolist(),
            },
        },
        train_loss=loss_tr_hp,
        holdout_rmse=rmse_ho_hp,
    ))

    # 2. Linear surrogate: sigma_y = a0 + a1 * d
    X_tr_lin = np.column_stack([np.ones_like(d_tr), d_tr])
    w_lin, *_ = np.linalg.lstsq(X_tr_lin, y_tr, rcond=None)
    pred_tr_lin = X_tr_lin @ w_lin
    loss_tr_lin = float(np.mean((y_tr - pred_tr_lin)**2))
    X_ho_lin = np.column_stack([np.ones_like(d_ho), d_ho])
    pred_ho_lin = X_ho_lin @ w_lin
    resid_lin = y_ho - pred_ho_lin
    rmse_ho_lin = float(np.sqrt(np.mean(resid_lin**2)))

    c_lin = CandidateDynamics(
        lhs="yield_strength",
        terms=(Term("one", "a0"), Term("d", "a1")),
        candidate_id="linear",
    )
    candidates.append(ScoredCandidate(
        c_lin,
        provenance={
            "data_origin": "simulation",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(d_ho),
                "rmse": rmse_ho_lin,
                "holdout_residuals": resid_lin.tolist(),
            },
        },
        train_loss=loss_tr_lin,
        holdout_rmse=rmse_ho_lin,
    ))

    # 3. Constant model: sigma_y = a0
    X_tr_c = np.column_stack([np.ones_like(d_tr)])
    w_c, *_ = np.linalg.lstsq(X_tr_c, y_tr, rcond=None)
    pred_tr_c = X_tr_c @ w_c
    loss_tr_c = float(np.mean((y_tr - pred_tr_c)**2))
    X_ho_c = np.column_stack([np.ones_like(d_ho)])
    pred_ho_c = X_ho_c @ w_c
    resid_c = y_ho - pred_ho_c
    rmse_ho_c = float(np.sqrt(np.mean(resid_c**2)))

    c_c = CandidateDynamics(
        lhs="yield_strength",
        terms=(Term("one", "a0"),),
        candidate_id="constant",
    )
    candidates.append(ScoredCandidate(
        c_c,
        provenance={
            "data_origin": "simulation",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(d_ho),
                "rmse": rmse_ho_c,
                "holdout_residuals": resid_c.tolist(),
            },
        },
        train_loss=loss_tr_c,
        holdout_rmse=rmse_ho_c,
    ))

    return candidates


# ── Run APEDP on Hall-Petch data ─────────────────────────────────────────
def run_hall_pecht_discovery(noise_sigma: float = 0.03,
                             bootstrap: bool = True,
                             data_origin: str = "simulation") -> tuple[Adjudication, np.ndarray, np.ndarray]:
    """Run APEDP on Hall-Petch data with genuinely fitted regression models."""
    d, sigma_y = generate_hall_pecht_data(noise_sigma=noise_sigma)
    candidates = fit_hall_pecht_candidates(d, sigma_y, n_train=20, noise_sigma=noise_sigma)

    adj = adjudicate(
        candidates,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        bootstrap=bootstrap,
        bootstrap_n_boot=1000,
        bootstrap_seed=42,
        data_origin=data_origin,
    )
    return adj, d, sigma_y


# ── Main demo execution ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("APEDP Hall-Petch Materials Science Strengthening Demo")
    print("=" * 65)

    for noise in (0.01, 0.03, 0.10):
        adj, d, y = run_hall_pecht_discovery(noise_sigma=noise, data_origin="simulation")
        print(f"\nNoise level: {noise*100:.0f}%")
        print(f"  Status:       {adj.status.value}")
        print(f"  Promoted:     {adj.promoted.candidate.candidate_id if adj.promoted else 'None'}")
        print(f"  Best RMSE:    {adj.best_rmse:.3f}" if adj.best_rmse is not None else "  Best RMSE: None")
        print(f"  Data Origin:  {adj.data_origin}")
        print(f"  Reason:       {adj.reason[:80]}...")
