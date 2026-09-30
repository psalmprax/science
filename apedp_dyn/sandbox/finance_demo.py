"""APEDP Financial Time Series Momentum Discovery Demo.

Demonstrates the APEDP epistemic adjudication pipeline on financial
momentum vs mean-reversion time series models using actually-fitted
autoregressive and trend candidates (no random losses).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

SOLE_SURVIVOR_GAMMA = 3.0
CAL_SIGMA = 1.8
RESID_RHO_FLAG = 0.5


# ── Synthetic financial data generator ──────────────────────────────────
def generate_finance_data(n_points: int = 100,
                          momentum: float = 0.05,
                          noise_sigma: float = 0.02,
                          regime: str = "bull") -> np.ndarray:
    """Generate synthetic price time series."""
    rng = np.random.RandomState(42)
    if regime == "bull":
        mom = momentum
    elif regime == "bear":
        mom = -abs(momentum)
    else:
        mom = 0.0

    prices = np.zeros(n_points)
    prices[0] = 100.0
    for t in range(1, n_points):
        prices[t] = prices[t - 1] * (1 + mom) + rng.normal(0, noise_sigma)
    return prices


# ── Fitted Candidate Generation ──────────────────────────────────────────
def fit_finance_candidates(prices: np.ndarray,
                           noise_sigma: float = 0.02,
                           data_origin: str = "simulation") -> list[ScoredCandidate]:
    """Fit actual AR(1), drift, and linear trend models on train split and score on holdout.

    Provenance data_origin can be 'simulation', 'synthetic', or 'real_observational'.
    """
    n = len(prices)
    y = prices[1:]
    x_lag = prices[:-1]
    t_idx = np.arange(1, n, dtype=float)

    n_samples = len(y)
    n_train = n_samples // 2

    y_tr, y_ho = y[:n_train], y[n_train:]
    x_lag_tr, x_lag_ho = x_lag[:n_train], x_lag[n_train:]
    t_tr, t_ho = t_idx[:n_train], t_idx[n_train:]

    obs_rms = float(np.sqrt(np.mean(y_ho**2)))
    candidates: list[ScoredCandidate] = []

    # H1: Autoregressive / Momentum: P_t = a0 + a1 * P_{t-1}
    X_tr_h1 = np.column_stack([np.ones_like(x_lag_tr), x_lag_tr])
    w_h1, *_ = np.linalg.lstsq(X_tr_h1, y_tr, rcond=None)
    pred_tr_h1 = X_tr_h1 @ w_h1
    loss_tr_h1 = float(np.mean((y_tr - pred_tr_h1)**2))
    X_ho_h1 = np.column_stack([np.ones_like(x_lag_ho), x_lag_ho])
    pred_ho_h1 = X_ho_h1 @ w_h1
    resid_h1 = y_ho - pred_ho_h1
    rmse_ho_h1 = float(np.sqrt(np.mean(resid_h1**2)))

    c_h1 = CandidateDynamics(
        lhs="price_t",
        terms=(Term("one", "a0"), Term("prev_price", "a1")),
        candidate_id="momentum_ar1",
    )
    candidates.append(ScoredCandidate(
        c_h1,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_sigma,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_h1,
                "holdout_residuals": resid_h1.tolist(),
            },
        },
        train_loss=loss_tr_h1,
        holdout_rmse=rmse_ho_h1,
    ))

    # H2: Constant price / No trend: P_t = a0
    X_tr_h2 = np.column_stack([np.ones_like(y_tr)])
    w_h2, *_ = np.linalg.lstsq(X_tr_h2, y_tr, rcond=None)
    pred_tr_h2 = X_tr_h2 @ w_h2
    loss_tr_h2 = float(np.mean((y_tr - pred_tr_h2)**2))
    X_ho_h2 = np.column_stack([np.ones_like(y_ho)])
    pred_ho_h2 = X_ho_h2 @ w_h2
    resid_h2 = y_ho - pred_ho_h2
    rmse_ho_h2 = float(np.sqrt(np.mean(resid_h2**2)))

    c_h2 = CandidateDynamics(
        lhs="price_t",
        terms=(Term("one", "a0"),),
        candidate_id="constant_price",
    )
    candidates.append(ScoredCandidate(
        c_h2,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_sigma,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_h2,
                "holdout_residuals": resid_h2.tolist(),
            },
        },
        train_loss=loss_tr_h2,
        holdout_rmse=rmse_ho_h2,
    ))

    # H3: Static Linear trend: P_t = a0 + a1 * t
    X_tr_h3 = np.column_stack([np.ones_like(t_tr), t_tr])
    w_h3, *_ = np.linalg.lstsq(X_tr_h3, y_tr, rcond=None)
    pred_tr_h3 = X_tr_h3 @ w_h3
    loss_tr_h3 = float(np.mean((y_tr - pred_tr_h3)**2))
    X_ho_h3 = np.column_stack([np.ones_like(t_ho), t_ho])
    pred_ho_h3 = X_ho_h3 @ w_h3
    resid_h3 = y_ho - pred_ho_h3
    rmse_ho_h3 = float(np.sqrt(np.mean(resid_h3**2)))

    c_h3 = CandidateDynamics(
        lhs="price_t",
        terms=(Term("one", "a0"), Term("t_index", "a1")),
        candidate_id="linear_trend",
    )
    candidates.append(ScoredCandidate(
        c_h3,
        provenance={
            "data_origin": data_origin,
            "gate4_status": {
                "passed": True,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_sigma,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho_h3,
                "holdout_residuals": resid_h3.tolist(),
            },
        },
        train_loss=loss_tr_h3,
        holdout_rmse=rmse_ho_h3,
    ))

    return candidates


# ── Run APEDP on finance data ──────────────────────────────────────────
def run_finance_discovery(prices: np.ndarray,
                          noise_sigma: float = 0.02,
                          bootstrap: bool = True,
                          data_origin: str = "simulation") -> Adjudication:
    """Run APEDP on financial data with fitted model candidates."""
    candidates = fit_finance_candidates(prices, noise_sigma=noise_sigma, data_origin=data_origin)
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


if __name__ == "__main__":
    print("=" * 65)
    print("APEDP Financial Momentum & Regime Discovery Demo")
    print("=" * 65)

    for regime, mom in [("bull", 0.05), ("bear", -0.05), ("sideways", 0.0)]:
        prices = generate_finance_data(n_points=100, momentum=mom, regime=regime)
        adj = run_finance_discovery(prices, noise_sigma=0.02, data_origin="simulation")
        print(f"\nRegime: {regime.upper()} (momentum={mom:.2f})")
        print(f"  Status:       {adj.status.value}")
        print(f"  Promoted:     {adj.promoted.candidate.candidate_id if adj.promoted else 'None'}")
        print(f"  Best RMSE:    {adj.best_rmse:.4f}" if adj.best_rmse is not None else "  Best RMSE: None")
        print(f"  Data Origin:  {adj.data_origin}")
        print(f"  Reason:       {adj.reason[:80]}...")
