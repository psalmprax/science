"""Dynamical Systems and Fluid Analogies Discovery Sweep across Multi-Asset Universes and Horizons.

Tests 5 explicit physical/dynamical hypotheses inspired by:
  1. OpenAI Mathematics repository (Navier-Stokes and Euler fluid dynamics)
  2. LLM-ACES repository (Autonomous, forced, and 2nd-order ODE state-evolution systems)

Evaluated across:
  - 4 Universes: 15, 30, 60, and 90 assets
  - 6 Timeframes / Horizons: 1m, 5m, 15m, 30m, 45m, 60m
  - 1-Month Bidirectional Rolling Validation: Train Month M, Forward Holdout M+1, Backward Holdout M-1
  - Dual Adjudication: APEDP Sovereign Epistemic Gated Arm vs. Ungated "Best-Fit" ML Arm

The 5 Explicit Equations:
  Eq 1 (Navier-Stokes Fluid Analogy):
    y = c1*lead_spill + c2*ofi + c3*disp_norm
    (Advection momentum + pressure force + viscous dispersion)

  Eq 2 (Incompressible Euler Flow Analogy):
    y = c1*lead_spill + c2*ofi
    (Inviscid convective momentum balance without viscous dissipation)

  Eq 3 (Autonomous State-Evolution ODE - LLM-ACES Eq 6):
    y = c1*rel_mom + c2*cs_rank_vol + c3*ofi
    (Linearized multi-state vector field: velocity + volatility drag + order flow)

  Eq 4 (Forced Non-Autonomous Dynamical System - LLM-ACES Eq 7):
    y = c1*lead_spill + c2*ofi + c3*(lead_spill * disp_norm)
    (State evolution modulated by external macro volatility context)

  Eq 5 (Second-Order Harmonic Acceleration ODE - LLM-ACES Eq 8):
    y = c1*rel_mom + c2*kalman_res + c3*delta_ofi
    (Acceleration d^2p/dt^2 driven by velocity momentum, restoring spring, and d(OFI)/dt)
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from apedp_dyn.discovery_l2.pruning import check_residual_whiteness
from apedp_dyn.epistemic.cross_asset_gate import evaluate_cross_asset_gate
from apedp_dyn.sandbox.grammar_space import causal_zscore, trailing_rank
from apedp_dyn.sandbox.multi_asset_signals_sandbox import load_universe

TIMEFRAMES_MINUTES = (1, 5, 15, 30, 45, 60)
UNIVERSE_SIZES = (15, 30, 60, 90)

CANDIDATE_EQUATION_SPECS: dict[str, dict[str, Any]] = {
    "Eq1_NavierStokes": {
        "name": "Eq 1: Navier-Stokes Fluid Analogy",
        "family": "Fluid Dynamics",
        "formula": "y = c1*lead_spill + c2*ofi + c3*disp_norm",
        "feature_keys": ["lead_spill", "ofi", "disp_norm"],
        "physical_interpretation": "Advective momentum transport + pressure gradient forcing + viscous dissipation",
    },
    "Eq2_Euler": {
        "name": "Eq 2: Incompressible Euler Flow",
        "family": "Fluid Dynamics",
        "formula": "y = c1*lead_spill + c2*ofi",
        "feature_keys": ["lead_spill", "ofi"],
        "physical_interpretation": "Inviscid convective momentum transport + order book pressure forcing",
    },
    "Eq3_ODE_Autonomous": {
        "name": "Eq 3: Autonomous State ODE (LLM-ACES Eq 6)",
        "family": "Dynamical Systems",
        "formula": "y = c1*rel_mom + c2*cs_rank_vol + c3*ofi",
        "feature_keys": ["rel_mom", "cs_rank_vol", "ofi"],
        "physical_interpretation": "Autonomous state evolution: relative momentum velocity + volatility drag + OFI push",
    },
    "Eq4_ODE_Forced": {
        "name": "Eq 4: Macro-Forced Dynamical System (LLM-ACES Eq 7)",
        "family": "Dynamical Systems",
        "formula": "y = c1*lead_spill + c2*ofi + c3*lead_spill_x_disp",
        "feature_keys": ["lead_spill", "ofi", "lead_spill_x_disp"],
        "physical_interpretation": "State evolution with bilinear macro context modulation F(x) + G(x)*z(t)",
    },
    "Eq5_ODE_2ndOrder": {
        "name": "Eq 5: Second-Order Harmonic Acceleration (LLM-ACES Eq 8)",
        "family": "Dynamical Systems",
        "formula": "y = c1*rel_mom + c2*kalman_res + c3*delta_ofi",
        "feature_keys": ["rel_mom", "kalman_res", "delta_ofi"],
        "physical_interpretation": "Second-order acceleration: velocity drag + restoring spring force + d(OFI)/dt",
    },
}


@dataclass(frozen=True)
class EquationEvaluation:
    eq_id: str
    name: str
    family: str
    formula: str
    fitted_formula: str
    feature_keys: tuple[str, ...]
    coefficients: dict[str, float]
    train_r2: float
    train_rmse: float
    train_bic: float
    fwd_r2: float
    fwd_rmse: float
    fwd_rho1: float
    fwd_whiteness_passed: bool
    bwd_r2: float
    bwd_rmse: float
    bwd_rho1: float
    bwd_whiteness_passed: bool
    condition_number: float
    gate4_fwd_passed: bool
    gate4_bwd_passed: bool
    epistemic_tier: str
    gated_verdict: str
    best_fit_verdict: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "eq_id": self.eq_id,
            "name": self.name,
            "family": self.family,
            "formula": self.formula,
            "fitted_formula": self.fitted_formula,
            "feature_keys": list(self.feature_keys),
            "coefficients": dict(self.coefficients),
            "train_r2": self.train_r2,
            "train_rmse": self.train_rmse,
            "train_bic": self.train_bic,
            "fwd_r2": self.fwd_r2,
            "fwd_rmse": self.fwd_rmse,
            "fwd_rho1": self.fwd_rho1,
            "fwd_whiteness_passed": self.fwd_whiteness_passed,
            "bwd_r2": self.bwd_r2,
            "bwd_rmse": self.bwd_rmse,
            "bwd_rho1": self.bwd_rho1,
            "bwd_whiteness_passed": self.bwd_whiteness_passed,
            "condition_number": self.condition_number,
            "gate4_fwd_passed": self.gate4_fwd_passed,
            "gate4_bwd_passed": self.gate4_bwd_passed,
            "epistemic_tier": self.epistemic_tier,
            "gated_verdict": self.gated_verdict,
            "best_fit_verdict": self.best_fit_verdict,
            "notes": self.notes,
        }


@dataclass
class UniverseHorizonExperimentResult:
    universe_size: int
    timeframe_minutes: int
    evaluations: dict[str, EquationEvaluation]
    gate_05_cang_passed: bool
    two_tier_sharpe: float
    winning_eq_id: str
    summary_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe_size": self.universe_size,
            "timeframe_minutes": self.timeframe_minutes,
            "gate_05_cang_passed": self.gate_05_cang_passed,
            "two_tier_sharpe": self.two_tier_sharpe,
            "winning_eq_id": self.winning_eq_id,
            "summary_verdict": self.summary_verdict,
            "evaluations": {k: v.to_dict() for k, v in self.evaluations.items()},
        }


def build_unified_feature_panel(
    universe: dict[str, pd.DataFrame],
    symbols: tuple[str, ...],
    timeframe_minutes: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Construct causally strictly aligned features for fluid and dynamical models.

    Returns ``(X_dict, y_target, X_asset, fwd_excess_2d)``. ``X_dict`` /
    ``y_target`` keep the historical cross-sectionally *averaged* aggregate
    panel. ``X_asset`` stacks the per-asset feature series (columns =
    ``symbols[1:]``; the lead asset is used only as the reference series) and
    ``fwd_excess_2d`` is the per-asset forward forward-excess matrix in the
    same column order, both consumed by the stratified CAN-G evaluation.
    """
    T = len(universe[symbols[0]])
    N = len(symbols)

    step = max(1, round(timeframe_minutes / 15.0))
    scale = math.sqrt(max(1.0, timeframe_minutes / 5.0))
    w_roll = max(5, round(20 * scale))

    rets_2d = np.zeros((T, N))
    prices_2d = np.zeros((T, N))
    ofis_2d = np.zeros((T, N))

    for i, s in enumerate(symbols):
        rets_2d[:, i] = universe[s]["r"].to_numpy(float)
        prices_2d[:, i] = universe[s]["price"].to_numpy(float)
        if "ofi" in universe[s].columns:
            ofis_2d[:, i] = universe[s]["ofi"].to_numpy(float)
        else:
            ofis_2d[:, i] = 0.0

    lead_ret = rets_2d[:, 0]
    lead_vol = pd.Series(lead_ret).rolling(w_roll, min_periods=5).std().ffill().to_numpy()
    lead_vol = np.maximum(lead_vol, 1e-4)

    cross_median = np.median(rets_2d, axis=1)
    cross_mean = np.mean(rets_2d, axis=1)
    cross_std = np.std(rets_2d, axis=1)
    cross_std = np.maximum(cross_std, 1e-4)

    # Forward return target (strictly aligned to forward cross-median, zero contemporaneous leakage)
    fwd_rets_2d = np.roll(rets_2d, -step, axis=0)
    fwd_rets_2d[-step:, :] = 0.0
    fwd_cross_median = np.median(fwd_rets_2d, axis=1)
    fwd_excess_target = np.mean(fwd_rets_2d - fwd_cross_median[:, None], axis=1)

    all_lead_spill = []
    all_ofi = []
    all_disp_norm = []
    all_rel_mom = []
    all_cs_rank_vol = []
    all_lead_spill_x_disp = []
    all_kalman_res = []
    all_delta_ofi = []

    for i in range(1, N):
        r_i = rets_2d[:, i]
        p_i = prices_2d[:, i]
        ofi_i = ofis_2d[:, i]

        # 1. Lead spillover
        cov_lead = pd.Series(r_i * lead_ret).rolling(w_roll, min_periods=5).mean().ffill().to_numpy()
        vol_i = pd.Series(r_i).rolling(w_roll, min_periods=5).std().ffill().to_numpy()
        vol_i = np.maximum(vol_i, 1e-4)
        corr_lead = cov_lead / np.maximum(vol_i * lead_vol, 1e-6)
        corr_lead = np.clip(corr_lead, -1.0, 1.0)
        lead_spill = (lead_ret / lead_vol) * corr_lead

        # 2. Normalized OFI
        z_ofi = causal_zscore(ofi_i)

        # 3. Normalized cross-sectional dispersion
        disp_norm = (r_i - cross_mean) / cross_std

        # 4. Relative momentum
        rel_mom = r_i - cross_median

        # 5. CS rank vs Vol rank
        rk_r = trailing_rank(r_i)
        rk_v = trailing_rank(vol_i)
        cs_rank_vol = rk_r - rk_v

        # 6. Interaction term: LeadSpill * DispNorm
        lead_spill_x_disp = lead_spill * disp_norm

        # 7. Cointegration / Kalman residual
        var_lead = pd.Series(lead_ret**2).rolling(w_roll, min_periods=5).mean().ffill().to_numpy()
        beta_i = cov_lead / np.maximum(var_lead, 1e-6)
        beta_i = np.clip(beta_i, -2.0, 3.0)
        spread = p_i - beta_i * prices_2d[:, 0]
        sp_mean = pd.Series(spread).rolling(w_roll, min_periods=5).mean().ffill().to_numpy()
        sp_std = pd.Series(spread).rolling(w_roll, min_periods=5).std().ffill().to_numpy()
        kalman_res = (spread - sp_mean) / np.maximum(sp_std, 1e-6)

        # 8. Delta OFI (time derivative of order flow)
        d_ofi = np.append(0.0, np.diff(ofi_i))
        z_d_ofi = causal_zscore(d_ofi)

        all_lead_spill.append(causal_zscore(lead_spill))
        all_ofi.append(z_ofi)
        all_disp_norm.append(disp_norm)
        all_rel_mom.append(rel_mom)
        all_cs_rank_vol.append(cs_rank_vol)
        all_lead_spill_x_disp.append(causal_zscore(lead_spill_x_disp))
        all_kalman_res.append(causal_zscore(kalman_res))
        all_delta_ofi.append(z_d_ofi)

    X_dict = {
        "lead_spill": np.nan_to_num(np.mean(all_lead_spill, axis=0), nan=0.0),
        "ofi": np.nan_to_num(np.mean(all_ofi, axis=0), nan=0.0),
        "disp_norm": np.nan_to_num(np.mean(all_disp_norm, axis=0), nan=0.0),
        "rel_mom": np.nan_to_num(np.mean(all_rel_mom, axis=0), nan=0.0),
        "cs_rank_vol": np.nan_to_num(np.mean(all_cs_rank_vol, axis=0), nan=0.0),
        "lead_spill_x_disp": np.nan_to_num(np.mean(all_lead_spill_x_disp, axis=0), nan=0.0),
        "kalman_res": np.nan_to_num(np.mean(all_kalman_res, axis=0), nan=0.0),
        "delta_ofi": np.nan_to_num(np.mean(all_delta_ofi, axis=0), nan=0.0),
    }
    y_target = np.nan_to_num(fwd_excess_target, nan=0.0)
    X_asset: dict[str, np.ndarray] = {
        "lead_spill": np.nan_to_num(np.stack(all_lead_spill, axis=1), nan=0.0),
        "ofi": np.nan_to_num(np.stack(all_ofi, axis=1), nan=0.0),
        "disp_norm": np.nan_to_num(np.stack(all_disp_norm, axis=1), nan=0.0),
        "rel_mom": np.nan_to_num(np.stack(all_rel_mom, axis=1), nan=0.0),
        "cs_rank_vol": np.nan_to_num(np.stack(all_cs_rank_vol, axis=1), nan=0.0),
        "lead_spill_x_disp": np.nan_to_num(np.stack(all_lead_spill_x_disp, axis=1), nan=0.0),
        "kalman_res": np.nan_to_num(np.stack(all_kalman_res, axis=1), nan=0.0),
        "delta_ofi": np.nan_to_num(np.stack(all_delta_ofi, axis=1), nan=0.0),
    }
    # Per-asset forward excess in the discovery target's own alignment; the
    # market (cross-median) leg is subtracted per row, so stratified L/S
    # portfolios built on these rows are market-neutral by construction.
    fwd_excess_2d = np.nan_to_num(fwd_rets_2d - fwd_cross_median[:, None], nan=0.0)
    return X_dict, y_target, X_asset, fwd_excess_2d


def evaluate_equation_epistemically(
    eq_id: str,
    spec: dict[str, Any],
    X_tr: dict[str, np.ndarray],
    y_tr: np.ndarray,
    X_fwd: dict[str, np.ndarray],
    y_fwd: np.ndarray,
    X_bwd: dict[str, np.ndarray],
    y_bwd: np.ndarray,
) -> EquationEvaluation:
    """Evaluate candidate equation with strict epistemic gates alongside naive best-fit baseline."""
    keys = spec["feature_keys"]
    k = len(keys)
    n_tr = len(y_tr)

    M_tr = np.column_stack([X_tr[k_i] for k_i in keys])
    M_fwd = np.column_stack([X_fwd[k_i] for k_i in keys])
    M_bwd = np.column_stack([X_bwd[k_i] for k_i in keys])

    # Condition number check
    norms = np.linalg.norm(M_tr, axis=0)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    M_scaled = M_tr / safe_norms
    s_vals = np.linalg.svd(M_scaled, compute_uv=False)
    cond = float(s_vals[0] / s_vals[-1]) if s_vals[-1] > 1e-15 else float("inf")

    # Fit in-sample OLS coefficients
    try:
        coefs, _, _, _ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
    except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
        coefs = np.zeros(k)

    # In-sample Train M metrics
    pred_tr = M_tr @ coefs
    res_tr = y_tr - pred_tr
    tr_mse = float(np.mean(res_tr**2))
    tr_rmse = float(np.sqrt(tr_mse))
    var_tr = float(np.var(y_tr))
    tr_r2 = float(1.0 - (tr_rmse**2) / (var_tr + 1e-12))
    tr_bic = float(n_tr * np.log(tr_mse + 1e-12) + k * np.log(n_tr))

    # Out-of-sample Forward Holdout M+1 metrics
    pred_fwd = M_fwd @ coefs
    res_fwd = y_fwd - pred_fwd
    fwd_rmse = float(np.sqrt(np.mean(res_fwd**2)))
    var_fwd = float(np.var(y_fwd))
    fwd_r2 = float(1.0 - (fwd_rmse**2) / (var_fwd + 1e-12))
    white_fwd, fwd_rho1 = check_residual_whiteness(res_fwd)

    # Out-of-sample Backward Holdout M-1 metrics
    pred_bwd = M_bwd @ coefs
    res_bwd = y_bwd - pred_bwd
    bwd_rmse = float(np.sqrt(np.mean(res_bwd**2)))
    var_bwd = float(np.var(y_bwd))
    bwd_r2 = float(1.0 - (bwd_rmse**2) / (var_bwd + 1e-12))
    white_bwd, bwd_rho1 = check_residual_whiteness(res_bwd)

    # Gate 4 checks (R2 >= 0.20, whiteness, conditioning)
    gate4_fwd = bool(fwd_r2 >= 0.20 and white_fwd and cond <= 1000.0)
    gate4_bwd = bool(bwd_r2 >= 0.20 and white_bwd and cond <= 1000.0)

    # APEDP Epistemic Tier Adjudication
    if gate4_fwd and gate4_bwd:
        tier = "C3_STRUCTURAL_INVARIANCE"
        gated_verdict = "PROMOTED_BIDIRECTIONAL"
    elif gate4_fwd:
        tier = "C2_PREDICTIVE_SUPPORT"
        gated_verdict = "PROMOTED_FORWARD_ONLY"
    elif tr_r2 >= 0.20:
        tier = "C1_DESCRIPTIVE"
        gated_verdict = "REFUSED_HOLDOUT_GATE_4"
    else:
        tier = "C0_NO_SUPPORT"
        gated_verdict = "TERMINAL_ABSTAIN"

    # Naive "Best-Fit" ML Arm Decision:
    # Standard ML has no epistemic abstention gate and always promotes the argmin OLS fit
    best_fit_verdict = "PROMOTES (Ungated Argmin)"

    coef_dict = {k_i: float(coefs[idx]) for idx, k_i in enumerate(keys)}
    terms_str = " + ".join([f"{coef_dict[k_i]:+.4f}*{k_i}" for k_i in keys])
    fitted_formula = f"y = {terms_str}"

    return EquationEvaluation(
        eq_id=eq_id,
        name=spec["name"],
        family=spec["family"],
        formula=spec["formula"],
        fitted_formula=fitted_formula,
        feature_keys=tuple(keys),
        coefficients=coef_dict,
        train_r2=tr_r2,
        train_rmse=tr_rmse,
        train_bic=tr_bic,
        fwd_r2=fwd_r2,
        fwd_rmse=fwd_rmse,
        fwd_rho1=fwd_rho1,
        fwd_whiteness_passed=white_fwd,
        bwd_r2=bwd_r2,
        bwd_rmse=bwd_rmse,
        bwd_rho1=bwd_rho1,
        bwd_whiteness_passed=white_bwd,
        condition_number=cond,
        gate4_fwd_passed=gate4_fwd,
        gate4_bwd_passed=gate4_bwd,
        epistemic_tier=tier,
        gated_verdict=gated_verdict,
        best_fit_verdict=best_fit_verdict,
        notes=spec["physical_interpretation"],
    )


def _cross_sectional_ls_pnl(scores_row: np.ndarray, excess_row: np.ndarray) -> float:
    """One-step market-neutral L/S P&L from a per-asset score row.

    The score row is z-scored across the stratum's assets (cross-sectional
    information only — no temporal leakage), then traded one-unit against
    the same-step forward excess. A constant score row (degenerate stratum)
    contributes zero P&L for that step.
    """
    scores_row = np.asarray(scores_row, dtype=float)
    excess_row = np.asarray(excess_row, dtype=float)
    if scores_row.size < 2:
        return 0.0
    sd = float(np.std(scores_row))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    z = (scores_row - float(np.mean(scores_row))) / sd
    return float(np.mean(z * excess_row))


def _sharpe(series: np.ndarray) -> float:
    """Per-period Sharpe of a P&L series (scale-free decision statistic)."""
    series = np.asarray(series, dtype=float)
    if series.size < 2:
        return 0.0
    sd = float(np.std(series, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(np.mean(series)) / sd


def evaluate_stratified_representation_gate(
    evals: dict[str, EquationEvaluation],
    winning_id: str,
    X_asset: dict[str, np.ndarray],
    fwd_excess_2d: np.ndarray,
    tr_slice: slice,
    sub_universe_count: int,
) -> tuple[bool, float]:
    """Gate 0.5 (CAN-G) evidence from the representation, not from raw market data.

    Historical defect: the probe fed the gate RAW per-asset returns, so the
    verdict measured the arbitrary market-drift sign of the window -- a
    coin flip -- and never touched the discovered representation at all.
    
    Now the gate evaluates the winning equation's FROZEN train-window
    coefficients as an actual stratified signal: per-asset scores are the
    per-asset feature panels (already causal z-scored / dispersion-
    normalized, so ``volatility_normalized=True`` becomes a true claim),
    scored against the same alignment the discovery target uses. Each
    sub-universe (positional quarter) yields a market-neutral L/S P&L
    series; the naive-pooled basket and the equal-allocation two-tier
    portfolio follow the reference multi-asset sandbox's semantics.

    Returns (passed, two_tier_sharpe). All steps use only training-window
    rows; the holdout referee sees none of this.
    """
    win_ev = evals[winning_id]
    panels = [X_asset[k] for k in win_ev.feature_keys]
    if any(p.ndim != 2 for p in panels) or len({p.shape for p in panels}) != 1:
        raise ValueError("per-asset feature panels must share shape (T, N-1)")

    rows = list(range(tr_slice.start, tr_slice.stop))
    if len(rows) < 3:
        raise ValueError("CAN-G probe needs a multi-row training window")

    coeffs = np.array([win_ev.coefficients[k] for k in win_ev.feature_keys], dtype=float)
    scores = np.zeros((len(rows), panels[0].shape[1]), dtype=float)
    for c, panel in zip(coeffs, panels, strict=True):
        scores += c * panel[rows, :]
    excess = fwd_excess_2d[rows, 1:]

    n_assets = scores.shape[1]
    n_sub = max(2, min(int(sub_universe_count), n_assets))
    bounds = [round(i * n_assets / n_sub) for i in range(n_sub + 1)]

    within: dict[str, float] = {}
    stratum_pnl: list[np.ndarray] = []
    for s in range(n_sub):
        pnl = np.array(
            [
                _cross_sectional_ls_pnl(scores[r, bounds[s]:bounds[s + 1]], excess[r, bounds[s]:bounds[s + 1]])
                for r in range(len(rows))
            ],
            dtype=float,
        )
        stratum_pnl.append(pnl)
        within[f"sub_universe_{s:02d}"] = _sharpe(pnl)

    pooled_pnl = np.array(
        [_cross_sectional_ls_pnl(scores[r, :], excess[r, :]) for r in range(len(rows))],
        dtype=float,
    )
    # Equal-RISK two-tier allocation (the gate module's "Equal-Risk
    # Allocation" semantic): each stratum leg is vol-targeted to one unit of
    # risk before aggregation, so a high-dispersion stratum cannot dominate
    # the combined portfolio merely by trading at a larger scale -- which is
    # precisely the cross-asset distortion CAN-G exists to catch.
    leg_risks = np.array(
        [float(np.std(p, ddof=1)) for p in stratum_pnl], dtype=float
    )
    if np.any(~np.isfinite(leg_risks)) or np.any(leg_risks <= 1e-12):
        # A degenerate (zero-dispersion) stratum cannot be evaluated
        # independently; the stratified evidence is inadmissible.
        return False, float("nan")
    two_tier_pnl = np.mean(
        np.stack([p / r for p, r in zip(stratum_pnl, leg_risks, strict=True)], axis=0),
        axis=0,
    )
    two_tier = _sharpe(two_tier_pnl)

    report = evaluate_cross_asset_gate(
        within_class_sharpes=within,
        naive_pooled_sharpe=_sharpe(pooled_pnl),
        two_tier_sharpe=two_tier,
        volatility_normalized=True,
        # Not a portfolio-promotion evaluation (discovery-phase cell): per
        # the gate module's doctrine, SR > 0 per class is required "ONLY
        # when the representation is being promoted as an active trading
        # strategy". Requiring it here failed good representations on
        # 21-row sign noise (each stratum Sharpe's noise scale is 1/sqrt(21)
        # ~ 0.22). The binding holdout decision stays with Gate 4.
        portfolio_promotion=False,
    )
    # Non-vacuous discovery-phase verdict: the stratified program must be
    # (a) scientifically valid, (b) fully evaluated in every stratum, and
    # (c) at least non-negative on the equal-risk two-tier allocation. A
    # negative or degenerate stratified program is itself cross-asset
    # evidence against the representation.
    if not report.passed or not report.scientific_valid:
        return False, two_tier
    if not np.isfinite(two_tier) or two_tier < 0.0:
        return False, two_tier
    return bool(report.passed), float(two_tier)


def run_single_universe_timeframe_experiment(
    universe_size: int = 30,
    timeframe_minutes: int = 15,
    month_start_idx: int = 80,
    month_len: int = 21,
) -> UniverseHorizonExperimentResult:
    """Run discovery on a single (universe_size, timeframe) cell for all 5 equations."""
    universe, symbols = load_universe(str(universe_size))
    X_dict, y_target, X_asset, fwd_excess_2d = build_unified_feature_panel(
        universe, symbols, timeframe_minutes
    )

    bwd_slice = slice(max(0, month_start_idx - month_len), month_start_idx)
    tr_slice = slice(month_start_idx, month_start_idx + month_len)
    fwd_slice = slice(month_start_idx + month_len, month_start_idx + 2 * month_len)

    X_bwd = {k: v[bwd_slice] for k, v in X_dict.items()}
    y_bwd = y_target[bwd_slice]
    X_tr = {k: v[tr_slice] for k, v in X_dict.items()}
    y_tr = y_target[tr_slice]
    X_fwd = {k: v[fwd_slice] for k, v in X_dict.items()}
    y_fwd = y_target[fwd_slice]

    evals = {}
    best_bic = float("inf")
    winning_id = "Eq1_NavierStokes"

    for eq_id, spec in CANDIDATE_EQUATION_SPECS.items():
        ev = evaluate_equation_epistemically(
            eq_id=eq_id,
            spec=spec,
            X_tr=X_tr,
            y_tr=y_tr,
            X_fwd=X_fwd,
            y_fwd=y_fwd,
            X_bwd=X_bwd,
            y_bwd=y_bwd,
        )
        evals[eq_id] = ev
        if ev.condition_number <= 1000.0 and ev.train_bic < best_bic:
            best_bic = ev.train_bic
            winning_id = eq_id

    # Gate 0.5 (CAN-G): representation-derived stratified evidence. The old
    # probe fed raw per-asset returns, so its verdict was the market-drift
    # sign of whichever 21-day window happened to be chosen (verified coin
    # flip: 27-29 passing windows out of 60, with zero representation
    # influence). It also failed OPEN on exceptions, fabricating a 1.25
    # Sharpe. The stratified evaluation below scores the discovered
    # representation itself and fails closed.
    try:
        cang_passed, sharpe = evaluate_stratified_representation_gate(
            evals=evals,
            winning_id=winning_id,
            X_asset=X_asset,
            fwd_excess_2d=fwd_excess_2d,
            tr_slice=tr_slice,
            sub_universe_count=min(universe_size, 4),
        )
        cang_passed = bool(cang_passed)
        sharpe = float(sharpe)
    except Exception:  # noqa: BLE001 - fail-closed probe: a failed gate diagnostic can never read as a pass
        cang_passed = False
        sharpe = float("nan")

    win_ev = evals[winning_id]
    summary_verdict = win_ev.gated_verdict

    return UniverseHorizonExperimentResult(
        universe_size=universe_size,
        timeframe_minutes=timeframe_minutes,
        evaluations=evals,
        gate_05_cang_passed=cang_passed,
        two_tier_sharpe=sharpe,
        winning_eq_id=winning_id,
        summary_verdict=summary_verdict,
    )


def run_full_fluid_ode_sweep(
    universe_sizes: tuple[int, ...] = UNIVERSE_SIZES,
    timeframes: tuple[int, ...] = TIMEFRAMES_MINUTES,
    month_start_idx: int = 80,
) -> list[UniverseHorizonExperimentResult]:
    """Execute complete 4x6=24 cell experiment grid testing all 5 equations."""
    results = []
    total = len(universe_sizes) * len(timeframes)
    count = 0
    t0 = time.time()

    print(f"Starting Fluid/ODE sweep: {len(universe_sizes)} universes x {len(timeframes)} horizons = {total} cells...")
    for u_size in universe_sizes:
        for tf in timeframes:
            count += 1
            res = run_single_universe_timeframe_experiment(
                universe_size=u_size,
                timeframe_minutes=tf,
                month_start_idx=month_start_idx,
            )
            results.append(res)
            print(f"[{count:02d}/{total:02d}] Universe={u_size:02d}, TF={tf:02d}m -> Winner: {res.winning_eq_id} | Gated: {res.summary_verdict}")

    dt = time.time() - t0
    print(f"Sweep complete in {dt:.2f}s ({total} experiment cells, {total * 5} equation evaluations).")
    return results
