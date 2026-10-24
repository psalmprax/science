"""APEDP Research Sandbox: Tokamak Plasma Parameter Space Exploration.

Demonstrates the APEDP epistemic adjudication pipeline on plasma energy
scaling laws, using actual regression fits on simulated tokamak observables
with native data_origin labeling and epistemic status reporting.
"""
from __future__ import annotations

import sys

sys.path.insert(0, '.')

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term
from apedp_dyn.sandbox.plasma_simulator import (
    generate_synthetic_dataset,
)

# Live-tree frozen constants
SOLE_SURVIVOR_GAMMA = 3.0
CAL_SIGMA = 1.8
RESID_RHO_FLAG = 0.5


def fit_plasma_candidates(observations: np.ndarray,
                          noise_sigma: float,
                          n_train: int = 250,
                          n_holdout: int = 150) -> list[ScoredCandidate]:
    """Fit actual scaling law candidates for fusion power P_fus against (T, n).

    Physical ground truth from simulator: P_fus = 1e-3 * n * T^2.
    """
    # Columns: 0: T_meas, 1: n_meas, 2: B_meas, 3: P_fus_meas
    T_all = observations[:, 0]
    n_all = observations[:, 1]
    y_all = observations[:, 3]

    # Split
    T_tr, T_ho = T_all[:n_train], T_all[n_train:n_train + n_holdout]
    n_tr, n_ho = n_all[:n_train], n_all[n_train:n_train + n_holdout]
    y_tr, y_ho = y_all[:n_train], y_all[n_train:n_train + n_holdout]

    obs_rms = float(np.sqrt(np.mean(y_ho**2)))
    noise_est = float(noise_sigma * obs_rms)
    candidates: list[ScoredCandidate] = []

    # Model 1: True law P_fus ~ a0 + a1 * (n * T^2)
    x_law_tr = n_tr * (T_tr**2)
    x_law_ho = n_ho * (T_ho**2)
    X_tr_1 = np.column_stack([np.ones_like(x_law_tr), x_law_tr])
    w_1, *_ = np.linalg.lstsq(X_tr_1, y_tr, rcond=None)
    pred_tr_1 = X_tr_1 @ w_1
    loss_tr_1 = float(np.mean((y_tr - pred_tr_1)**2))
    X_ho_1 = np.column_stack([np.ones_like(x_law_ho), x_law_ho])
    pred_ho_1 = X_ho_1 @ w_1
    resid_1 = y_ho - pred_ho_1
    rmse_ho_1 = float(np.sqrt(np.mean(resid_1**2)))

    c1 = CandidateDynamics(
        lhs="P_fus",
        terms=(Term("one", "a0"), Term("n_T_sq", "a1")),
        candidate_id="lawson_true_nT2",
    )
    candidates.append(ScoredCandidate(
        c1,
        provenance={
            "data_origin": "simulation",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_1,
                "holdout_residuals": resid_1.tolist(),
            },
        },
        train_loss=loss_tr_1,
        holdout_rmse=rmse_ho_1,
    ))

    # Model 2: Linear approximation P_fus ~ a0 + a1*T + a2*n
    X_tr_2 = np.column_stack([np.ones_like(T_tr), T_tr, n_tr])
    w_2, *_ = np.linalg.lstsq(X_tr_2, y_tr, rcond=None)
    pred_tr_2 = X_tr_2 @ w_2
    loss_tr_2 = float(np.mean((y_tr - pred_tr_2)**2))
    X_ho_2 = np.column_stack([np.ones_like(T_ho), T_ho, n_ho])
    pred_ho_2 = X_ho_2 @ w_2
    resid_2 = y_ho - pred_ho_2
    rmse_ho_2 = float(np.sqrt(np.mean(resid_2**2)))

    c2 = CandidateDynamics(
        lhs="P_fus",
        terms=(Term("one", "a0"), Term("T", "a1"), Term("n", "a2")),
        candidate_id="linear_Tn",
    )
    candidates.append(ScoredCandidate(
        c2,
        provenance={
            "data_origin": "simulation",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_2,
                "holdout_residuals": resid_2.tolist(),
            },
        },
        train_loss=loss_tr_2,
        holdout_rmse=rmse_ho_2,
    ))

    # Model 3: Temperature-only quadratic P_fus ~ a0 + a1*T^2
    X_tr_3 = np.column_stack([np.ones_like(T_tr), T_tr**2])
    w_3, *_ = np.linalg.lstsq(X_tr_3, y_tr, rcond=None)
    pred_tr_3 = X_tr_3 @ w_3
    loss_tr_3 = float(np.mean((y_tr - pred_tr_3)**2))
    X_ho_3 = np.column_stack([np.ones_like(T_ho), T_ho**2])
    pred_ho_3 = X_ho_3 @ w_3
    resid_3 = y_ho - pred_ho_3
    rmse_ho_3 = float(np.sqrt(np.mean(resid_3**2)))

    c3 = CandidateDynamics(
        lhs="P_fus",
        terms=(Term("one", "a0"), Term("T_sq", "a1")),
        candidate_id="quadratic_T_only",
    )
    candidates.append(ScoredCandidate(
        c3,
        provenance={
            "data_origin": "simulation",
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_3,
                "holdout_residuals": resid_3.tolist(),
            },
        },
        train_loss=loss_tr_3,
        holdout_rmse=rmse_ho_3,
    ))

    return candidates


def run_apedp_discovery(observations: np.ndarray,
                        noise_sigma: float,
                        bootstrap: bool = True,
                        data_origin: str = "simulation") -> Adjudication:
    """Run the APEDP adjudication pipeline with real fitted candidate hypotheses."""
    candidates = fit_plasma_candidates(observations, noise_sigma=noise_sigma)
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
    return adj


def main():
    print("=" * 65)
    print("APEDP Tokamak Plasma Confinement & Scaling Discovery Demo")
    print("=" * 65)

    datasets = generate_synthetic_dataset(n_conditions=15, noise_levels=(0.03, 0.05, 0.10))

    for _key, ds_info in datasets.items():
        noise = ds_info["noise_sigma"]
        obs = ds_info["observations"]
        adj = run_apedp_discovery(obs, noise_sigma=noise, data_origin="simulation")

        print(f"\n--- Noise level: {noise*100:.0f}% ---")
        print(f"  Status:       {adj.status.value}")
        print(f"  Promoted:     {adj.promoted.candidate.candidate_id if adj.promoted else 'None'}")
        print(f"  Best RMSE:    {adj.best_rmse:.6f}" if adj.best_rmse is not None else "  Best RMSE: None")
        print(f"  Data Origin:  {adj.data_origin}")
        print(f"  Reason:       {adj.reason[:80]}...")


if __name__ == "__main__":
    main()
