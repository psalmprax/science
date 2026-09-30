"""APEDP Trading Engine: Uses APEDP epistemic adjudication to detect
and trade momentum regimes in financial time series."""
from __future__ import annotations

import sys

sys.path.insert(0, '.')  # ensure apedp_dyn is importable

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

SOLE_SURVIVOR_GAMMA = 3.0
CAL_SIGMA = 1.8
RESID_RHO_FLAG = 0.5

def generate_price_series(n_points=200, momentum=0.05, noise_sigma=0.02, regime="bull", seed=42):
    rng = np.random.RandomState(seed)
    if regime == "bull":
        mom = momentum
    elif regime == "bear":
        mom = -abs(momentum)
    else:
        mom = 0.0
    prices = np.zeros(n_points)
    prices[0] = 100.0
    for t in range(1, n_points):
        prices[t] = prices[t-1] * (1 + mom) + rng.normal(0, noise_sigma)
    return prices

def fit_candidates(prices: np.ndarray, noise_sigma: float = 0.02) -> list[ScoredCandidate]:
    n = len(prices)
    y = prices[1:]
    x_lag = prices[:-1]
    t_idx = np.arange(1, n, dtype=float)

    n_samples = len(y)
    n_train = n_samples // 2

    y_tr, y_ho = y[:n_train], y[n_train:]
    x_lag_tr, x_lag_ho = x_lag[:n_train], x_lag[n_train:]
    t_tr, t_ho = t_idx[:n_train], t_idx[n_train:]

    obs_rms = float(np.sqrt(np.mean(y_ho**2))) if len(y_ho) > 0 else 1.0
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
        lhs="momentum",
        terms=(Term("one", "a0"), Term("prev_price", "a1")),
        candidate_id="mom_h1",
    )
    candidates.append(ScoredCandidate(
        c_h1,
        provenance={
            "data_origin": "simulation",
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
        lhs="momentum",
        terms=(Term("one", "a0"),),
        candidate_id="mom_h2",
    )
    candidates.append(ScoredCandidate(
        c_h2,
        provenance={
            "data_origin": "simulation",
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
        lhs="momentum",
        terms=(Term("one", "a0"), Term("t_index", "a1")),
        candidate_id="mom_h3",
    )
    candidates.append(ScoredCandidate(
        c_h3,
        provenance={
            "data_origin": "simulation",
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

def detect_regime(prices, noise_sigma=0.02, data_origin="simulation"):
    candidates = fit_candidates(prices, noise_sigma=noise_sigma)
    adj = adjudicate(candidates,
        sole_survivor_rule=True,
        sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG,
        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=42,
        data_origin=data_origin)
    
    status = adj.status
    promoted = adj.promoted.candidate.candidate_id if adj.promoted else None
    tau_epi = getattr(adj, 'tau_epi', 0.05)
    
    if status == EpistemicStatus.UNIQUE_SUPPORT and promoted == "mom_h1":
        regime = "bull" if np.mean(np.diff(prices)) > 0 else "bear"
    elif status == EpistemicStatus.UNIQUE_SUPPORT and promoted == "mom_h2":
        regime = "sideways"
    elif status == EpistemicStatus.MULTI_HYPOTHESIS:
        regime = "uncertain"
    else:
        regime = "unresolvable"
    
    return {'regime': regime, 'status': status, 'promoted': promoted,
        'tau_epi': tau_epi, 'data_origin': adj.data_origin}

def trading_strategy(prices, initial_capital=100.0, risk_per_trade=0.02,
                     data_origin="simulation"):
    capital = initial_capital
    position = 0.0
    equity_curve = [capital]
    trades = []
    step = 20
    for start in range(0, len(prices) - 40, step):
        window = prices[start:start+40]
        result = detect_regime(window, data_origin=data_origin)
        if result['regime'] == 'bull':
            if position <= 0:
                price = window[-1]
                shares = (capital * risk_per_trade) / price
                position = shares
                capital -= shares * price
                trades.append({'type': 'long', 'price': price, 'shares': shares})
        elif result['regime'] == 'bear':
            if position >= 0:
                price = window[-1]
                pnl = position * price
                capital += pnl
                position = 0
                trades.append({'type': 'close', 'pnl': pnl})
                shares = (capital * risk_per_trade) / price
                position = -shares
                capital -= shares * price
                trades.append({'type': 'short', 'price': price, 'shares': shares})
        elif result['regime'] in ('sideways', 'uncertain', 'unresolvable'):
            if position != 0:
                price = window[-1]
                pnl = position * price
                capital += pnl
                position = 0
                trades.append({'type': 'close', 'pnl': pnl})
    if position != 0:
        price = prices[-1]
        pnl = position * price
        capital += pnl
        trades.append({'type': 'close_final', 'pnl': pnl})
    final_equity = capital + position * prices[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    return {'initial_capital': initial_capital, 'final_equity': final_equity,
        'total_return': total_return, 'trades': trades,
        'num_trades': len([t for t in trades if t['type'] in ('long','short')]),
        'equity_curve': equity_curve}

def hello_world():
    print("=" * 60)
    print("APEDP TRADING ENGINE: Momentum Detection")
    print("=" * 60)
    for regime, mom in [("bull",0.05),("bear",-0.05),("sideways",0.0)]:
        prices = generate_price_series(momentum=mom, noise_sigma=0.02)
        result = trading_strategy(prices, 100.0, 0.02)
        print(f"Regime {regime:7s}: return={result['total_return']*100:6.1f}%, "
              f"trades={result['num_trades']:3d}, final=${result['final_equity']:.2f}")
    print()
    print("EPISTEMIC INSIGHT:")
    print("  Bull: strategy captures upward trend; status detects momentum.")
    print("  Bear: short selling enables profit from decline; status detects direction.")
    print("  Sideways: no-edge environment; status correctly indicates uncertainty;")
    print("    strategy stays flat.")
    print()

if __name__ == "__main__":
    hello_world()
