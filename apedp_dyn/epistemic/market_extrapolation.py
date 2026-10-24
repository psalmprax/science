"""APEDP-X Market Extrapolation Ladder (ESEDS for Financial Markets).

Bridges Domain A (Epistemically Safe Extrapolation of Dynamical Systems - ESEDS)
into quantitative trading and multi-asset portfolio sandboxes.

The 6-Tier Market Extrapolation Ladder:
    [Empirical Generalization Rungs]
    M0: In-Sample Calibration (Baseline parameter fit on calibration window)
    M1: Short Forward Holdout (Unseen sequential forward interval in-regime)
    M2: Extended Horizon Drift (Extended out-of-sample testing alpha decay & turnover drag)

    [Controlled Counterfactual Stress Rungs]
    M3: Volatility Shock Extrapolation (2.5x volatility expansion / ATR surge)
    M4: Liquidity Drought / Slippage Blowup (5x-10x fees & bid-ask spread expansion)
    M5: Structural Regime Crash & Correlation Coupling (Crisis co-movement / bear trend inversion)

Core Epistemic Principles:
    1. In-Sample Fit (M0) =/=> Forward Generalization (M1-M2) =/=> Stress Survival (M3-M5).
    2. Empirical Generalization vs Counterfactual Stress:
       M1-M2 constitute empirical evidence of temporal out-of-sample transfer.
       M3-M5 constitute controlled counterfactual stress tests under predefined shock parameters.
       Survival of M3-M5 demonstrates *conditional survival under specified synthetic stress*,
       NOT an unconstrained scientific claim of universal "structural crisis invariance."
    3. Observable Performance Signal vs Epistemic Diagnosis:
       A negative Sharpe ratio or drawdown threshold breach serves as an observable
       performance failure signal that triggers diagnostic analysis, rather than
       constituting the epistemic diagnosis in isolation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RungType(str, Enum):
    """Ontological classification of extrapolation rungs."""
    EMPIRICAL_GENERALIZATION = "empirical_generalization"
    CONTROLLED_COUNTERFACTUAL_STRESS = "controlled_counterfactual_stress"


class MarketRung(str, Enum):
    """The 6 rungs of the Market Extrapolation Ladder."""
    M0_IN_SAMPLE = "M0_in_sample"
    M1_FORWARD_HOLDOUT = "M1_forward_holdout"
    M2_EXTENDED_DRIFT = "M2_extended_drift"
    M3_VOLATILITY_SHOCK = "M3_volatility_shock"
    M4_LIQUIDITY_DROUGHT = "M4_liquidity_drought"
    M5_STRUCTURAL_CRASH = "M5_structural_crash"

    def label(self) -> str:
        names = {
            MarketRung.M0_IN_SAMPLE: "M0: In-Sample Calibration",
            MarketRung.M1_FORWARD_HOLDOUT: "M1: Short Forward Holdout",
            MarketRung.M2_EXTENDED_DRIFT: "M2: Extended Horizon Drift",
            MarketRung.M3_VOLATILITY_SHOCK: "M3: Volatility Shock (2.5x ATR)",
            MarketRung.M4_LIQUIDITY_DROUGHT: "M4: Liquidity Drought (5x Slippage)",
            MarketRung.M5_STRUCTURAL_CRASH: "M5: Structural Crisis & Coupling",
        }
        return names.get(self, str(self.name))

    def rung_type(self) -> RungType:
        if self in (MarketRung.M0_IN_SAMPLE, MarketRung.M1_FORWARD_HOLDOUT, MarketRung.M2_EXTENDED_DRIFT):
            return RungType.EMPIRICAL_GENERALIZATION
        return RungType.CONTROLLED_COUNTERFACTUAL_STRESS


@dataclass
class RungEvaluation:
    """Metrics and status for a single rung of the Market Extrapolation Ladder."""
    rung: MarketRung
    sharpe: float
    total_return_pct: float
    max_dd: float
    win_rate: float
    trades: int
    passed: bool
    notes: str = ""
    rung_type: str = "empirical_generalization"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung.value,
            "label": self.rung.label(),
            "rung_type": self.rung_type,
            "sharpe": float(round(self.sharpe, 2)),
            "total_return_pct": float(round(self.total_return_pct, 2)),
            "max_dd": float(round(self.max_dd, 1)),
            "win_rate": float(round(self.win_rate, 1)),
            "trades": int(self.trades),
            "passed": bool(self.passed),
            "notes": self.notes,
        }


@dataclass
class MarketExtrapolationResult:
    """Complete evaluation report across all 6 rungs of the Market Extrapolation Ladder."""
    candidate_id: str
    rungs: dict[str, RungEvaluation] = field(default_factory=dict)
    highest_rung_passed: MarketRung = MarketRung.M0_IN_SAMPLE
    extrapolative_safe: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    vol_shock_survival_ratio: float = 1.0
    liquidity_drag_ratio: float = 1.0
    structural_retention_ratio: float = 1.0
    empirical_transfer_passed: bool = False
    counterfactual_stress_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rungs": {k: v.to_dict() for k, v in self.rungs.items()},
            "highest_rung_passed": self.highest_rung_passed.value,
            "highest_rung_label": self.highest_rung_passed.label(),
            "extrapolative_safe": bool(self.extrapolative_safe),
            "empirical_transfer_passed": bool(self.empirical_transfer_passed),
            "counterfactual_stress_passed": bool(self.counterfactual_stress_passed),
            "failure_reasons": list(self.failure_reasons),
            "vol_shock_survival_ratio": float(round(self.vol_shock_survival_ratio, 2)),
            "liquidity_drag_ratio": float(round(self.liquidity_drag_ratio, 2)),
            "structural_retention_ratio": float(round(self.structural_retention_ratio, 2)),
        }


def evaluate_single_asset_market_ladder(
    candidate_id: str,
    simulate_fn: Callable[..., tuple[float, float, float, float, int, float, float, float]],
    scores: np.ndarray,
    prices: np.ndarray,
    train_end_idx: int,
    initial_capital: float = 10000.0,
    threshold: float = 0.10,
    base_fee_bps: float = 5.0,
    max_acceptable_dd: float = 40.0,
) -> MarketExtrapolationResult:
    """Evaluate a single-asset signal model across the 6-tier Market Extrapolation Ladder.

    Simulate_fn signature:
        simulate_fn(scores, prices, threshold=..., fee_bps=..., initial_capital=...)
        -> (total_return_pct, sharpe, acc, pnl, trades, pf, max_dd, win_rate)
    """
    total_len = min(len(scores), len(prices))
    n_tr = train_end_idx
    n_ho = total_len - n_tr
    n_mid = n_tr + (n_ho // 2)

    sc_tr = scores[:n_tr]
    px_tr = prices[:n_tr]

    sc_m1 = scores[n_tr:n_mid]
    px_m1 = prices[n_tr:n_mid]

    sc_m2 = scores[n_mid:total_len]
    px_m2 = prices[n_mid:total_len]

    failures: list[str] = []
    rungs: dict[str, RungEvaluation] = {}

    # M0: In-Sample Calibration
    r0_ret, r0_sr, _, _, r0_tr, _, r0_dd, r0_wr = simulate_fn(
        sc_tr, px_tr, threshold=threshold, fee_bps=base_fee_bps, initial_capital=initial_capital
    )
    m0_pass = bool(r0_sr >= 0.0 and r0_dd <= max_acceptable_dd)
    if not m0_pass:
        failures.append("M0_calibration_failed_in_sample_unstable")
    rungs["M0"] = RungEvaluation(
        rung=MarketRung.M0_IN_SAMPLE,
        sharpe=r0_sr,
        total_return_pct=r0_ret,
        max_dd=r0_dd,
        win_rate=r0_wr,
        trades=r0_tr,
        passed=m0_pass,
        notes="Calibration window fit",
    )

    # M1: Short Forward Holdout
    r1_ret, r1_sr, _, _, r1_tr, _, r1_dd, r1_wr = simulate_fn(
        sc_m1, px_m1, threshold=threshold, fee_bps=base_fee_bps, initial_capital=initial_capital
    )
    m1_pass = bool(m0_pass and (r1_sr >= -0.2 and r1_dd <= max_acceptable_dd))
    if m0_pass and not m1_pass:
        failures.append("M1_forward_holdout_immediate_decay")
    rungs["M1"] = RungEvaluation(
        rung=MarketRung.M1_FORWARD_HOLDOUT,
        sharpe=r1_sr,
        total_return_pct=r1_ret,
        max_dd=r1_dd,
        win_rate=r1_wr,
        trades=r1_tr,
        passed=m1_pass,
        notes="Sequential forward in-regime holdout",
    )

    # M2: Extended Horizon Drift
    r2_ret, r2_sr, _, _, r2_tr, _, r2_dd, r2_wr = simulate_fn(
        sc_m2, px_m2, threshold=threshold, fee_bps=base_fee_bps, initial_capital=initial_capital
    )
    m2_pass = bool(m1_pass and (r2_dd <= max_acceptable_dd + 5.0))
    if m1_pass and not m2_pass:
        failures.append("M2_extended_drift_excessive_decay")
    rungs["M2"] = RungEvaluation(
        rung=MarketRung.M2_EXTENDED_DRIFT,
        sharpe=r2_sr,
        total_return_pct=r2_ret,
        max_dd=r2_dd,
        win_rate=r2_wr,
        trades=r2_tr,
        passed=m2_pass,
        notes="Extended out-of-sample drift horizon",
    )

    # M3: Volatility Shock Extrapolation (2.5x price swing volatility on holdout)
    # Synthesize volatility expansion: perturb prices with 2.5x high-frequency fluctuation
    px_ho_full = prices[n_tr:total_len].copy()
    sc_ho_full = scores[n_tr:total_len].copy()
    ret_ho = np.diff(px_ho_full) / np.maximum(px_ho_full[:-1], 1e-12)
    shock_ret = ret_ho * 2.5  # 2.5x daily return dispersion
    px_shock = np.zeros(len(px_ho_full))
    px_shock[0] = px_ho_full[0]
    for i in range(len(shock_ret)):
        px_shock[i + 1] = max(px_shock[i] * (1.0 + shock_ret[i]), 1e-4)

    r3_ret, r3_sr, _, _, r3_tr, _, r3_dd, r3_wr = simulate_fn(
        sc_ho_full, px_shock, threshold=threshold, fee_bps=base_fee_bps, initial_capital=initial_capital
    )
    m3_pass = bool(m2_pass and (r3_dd <= 45.0))
    if m2_pass and not m3_pass:
        failures.append(f"M3_volatility_shock_divergence_drawdown_{r3_dd:.1f}pct")
    rungs["M3"] = RungEvaluation(
        rung=MarketRung.M3_VOLATILITY_SHOCK,
        sharpe=r3_sr,
        total_return_pct=r3_ret,
        max_dd=r3_dd,
        win_rate=r3_wr,
        trades=r3_tr,
        passed=m3_pass,
        notes="2.5x ATR volatility shock stress test",
    )

    # M4: Liquidity Drought / Slippage Blowup (5x fee/slippage friction: 25 bps)
    r4_ret, r4_sr, _, _, r4_tr, _, r4_dd, r4_wr = simulate_fn(
        sc_ho_full, px_ho_full, threshold=threshold, fee_bps=base_fee_bps * 5.0, initial_capital=initial_capital
    )
    m4_pass = bool(m2_pass and (r4_dd <= 45.0 and r4_ret >= -35.0))
    if m2_pass and not m4_pass:
        failures.append(f"M4_liquidity_drought_friction_collapse_{r4_ret:.1f}pct")
    rungs["M4"] = RungEvaluation(
        rung=MarketRung.M4_LIQUIDITY_DROUGHT,
        sharpe=r4_sr,
        total_return_pct=r4_ret,
        max_dd=r4_dd,
        win_rate=r4_wr,
        trades=r4_tr,
        passed=m4_pass,
        notes="5x transaction friction / slippage surge",
    )

    # M5: Structural Regime Crash (Simulate bear crash regime: -25% sustained trend drift)
    crash_ret = ret_ho - 0.005  # Systematic downward drag
    px_crash = np.zeros(len(px_ho_full))
    px_crash[0] = px_ho_full[0]
    for i in range(len(crash_ret)):
        px_crash[i + 1] = max(px_crash[i] * (1.0 + crash_ret[i]), 1e-4)

    r5_ret, r5_sr, _, _, r5_tr, _, r5_dd, r5_wr = simulate_fn(
        sc_ho_full, px_crash, threshold=threshold, fee_bps=base_fee_bps, initial_capital=initial_capital
    )
    m5_pass = bool(m3_pass and m4_pass and (r5_dd <= 45.0))
    if (m3_pass and m4_pass) and not m5_pass:
        failures.append(f"M5_structural_crash_drawdown_catastrophe_{r5_dd:.1f}pct")
    rungs["M5"] = RungEvaluation(
        rung=MarketRung.M5_STRUCTURAL_CRASH,
        sharpe=r5_sr,
        total_return_pct=r5_ret,
        max_dd=r5_dd,
        win_rate=r5_wr,
        trades=r5_tr,
        passed=m5_pass,
        notes="Structural bear regime drift inversion",
    )

    # Determine highest rung passed
    highest = MarketRung.M0_IN_SAMPLE
    order = [
        MarketRung.M0_IN_SAMPLE,
        MarketRung.M1_FORWARD_HOLDOUT,
        MarketRung.M2_EXTENDED_DRIFT,
        MarketRung.M3_VOLATILITY_SHOCK,
        MarketRung.M4_LIQUIDITY_DROUGHT,
        MarketRung.M5_STRUCTURAL_CRASH,
    ]
    for r in order:
        r_key = r.value.split("_")[0].upper()
        if rungs.get(r_key, RungEvaluation(r, 0, 0, 0, 0, 0, False)).passed:
            highest = r
        else:
            break

    extrap_safe = (highest == MarketRung.M5_STRUCTURAL_CRASH)
    vol_ratio = (r3_dd / max(r1_dd, 1.0)) if r1_dd > 0 else 1.0
    liq_ratio = (abs(r4_ret) / max(abs(r1_ret), 1.0)) if abs(r1_ret) > 0 else 1.0

    empirical_passed = bool(
        rungs.get("M0", RungEvaluation(MarketRung.M0_IN_SAMPLE, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M1", RungEvaluation(MarketRung.M1_FORWARD_HOLDOUT, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M2", RungEvaluation(MarketRung.M2_EXTENDED_DRIFT, 0, 0, 0, 0, 0, False)).passed
    )
    counterfactual_passed = bool(
        rungs.get("M3", RungEvaluation(MarketRung.M3_VOLATILITY_SHOCK, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M4", RungEvaluation(MarketRung.M4_LIQUIDITY_DROUGHT, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M5", RungEvaluation(MarketRung.M5_STRUCTURAL_CRASH, 0, 0, 0, 0, 0, False)).passed
    )

    return MarketExtrapolationResult(
        candidate_id=candidate_id,
        rungs=rungs,
        highest_rung_passed=highest,
        extrapolative_safe=extrap_safe,
        failure_reasons=failures,
        vol_shock_survival_ratio=vol_ratio,
        liquidity_drag_ratio=liq_ratio,
        structural_retention_ratio=1.0 if extrap_safe else 0.5,
        empirical_transfer_passed=empirical_passed,
        counterfactual_stress_passed=counterfactual_passed,
    )


def evaluate_30_asset_market_ladder(
    candidate_id: str,
    simulate_portfolio_fn: Callable[..., tuple[float, float, float, float, int, float, float, float]],
    scores_3d: np.ndarray,
    returns_2d: np.ndarray,
    vols_2d: np.ndarray,
    threshold: float = 0.10,
    base_fee_bps: float = 5.0,
    initial_capital: float = 100000.0,
    mode: str = "cross_sectional_ls",
    sizing: str = "risk_parity",
) -> MarketExtrapolationResult:
    """Evaluate 30-asset portfolio allocation across the 6-tier Market Extrapolation Ladder.

    Simulate_portfolio_fn signature:
        simulate_portfolio_fn(scores_3d, returns_2d, vols_2d, mode=..., sizing=..., threshold=..., initial_capital=...)
        -> (total_return_pct, sharpe, acc, pnl, trades, pf, max_dd, win_rate)
    """
    T, _N = scores_3d.shape
    n_tr = T // 2
    n_mid = n_tr + (T - n_tr) // 2

    # Slices
    s_tr, r_tr, v_tr = scores_3d[:n_tr], returns_2d[:n_tr], vols_2d[:n_tr]
    s_m1, r_m1, v_m1 = scores_3d[n_tr:n_mid], returns_2d[n_tr:n_mid], vols_2d[n_tr:n_mid]
    s_m2, r_m2, v_m2 = scores_3d[n_mid:], returns_2d[n_mid:], vols_2d[n_mid:]
    s_ho, r_ho, v_ho = scores_3d[n_tr:], returns_2d[n_tr:], vols_2d[n_tr:]

    failures: list[str] = []
    rungs: dict[str, RungEvaluation] = {}

    # M0: In-Sample Calibration
    r0_ret, r0_sr, _, _, r0_tr, _, r0_dd, r0_wr = simulate_portfolio_fn(
        s_tr, r_tr, v_tr, mode=mode, sizing=sizing, threshold=threshold, initial_capital=initial_capital
    )
    m0_pass = bool(r0_sr >= -0.5 and r0_dd <= 45.0)
    if not m0_pass:
        failures.append("M0_30_asset_in_sample_calibration_failed")
    rungs["M0"] = RungEvaluation(
        rung=MarketRung.M0_IN_SAMPLE,
        sharpe=r0_sr,
        total_return_pct=r0_ret,
        max_dd=r0_dd,
        win_rate=r0_wr,
        trades=r0_tr,
        passed=m0_pass,
        notes="30-asset in-sample calibration window",
    )

    # M1: Short Forward Holdout
    r1_ret, r1_sr, _, _, r1_tr, _, r1_dd, r1_wr = simulate_portfolio_fn(
        s_m1, r_m1, v_m1, mode=mode, sizing=sizing, threshold=threshold, initial_capital=initial_capital
    )
    m1_pass = bool(m0_pass and (r1_dd <= 45.0))
    if m0_pass and not m1_pass:
        failures.append("M1_30_asset_forward_holdout_breakdown")
    rungs["M1"] = RungEvaluation(
        rung=MarketRung.M1_FORWARD_HOLDOUT,
        sharpe=r1_sr,
        total_return_pct=r1_ret,
        max_dd=r1_dd,
        win_rate=r1_wr,
        trades=r1_tr,
        passed=m1_pass,
        notes="Sequential forward slice (15 crypto + 15 equities)",
    )

    # M2: Extended Horizon Drift
    r2_ret, r2_sr, _, _, r2_tr, _, r2_dd, r2_wr = simulate_portfolio_fn(
        s_m2, r_m2, v_m2, mode=mode, sizing=sizing, threshold=threshold, initial_capital=initial_capital
    )
    m2_pass = bool(m1_pass and (r2_dd <= 50.0))
    if m1_pass and not m2_pass:
        failures.append("M2_30_asset_extended_drift_decay")
    rungs["M2"] = RungEvaluation(
        rung=MarketRung.M2_EXTENDED_DRIFT,
        sharpe=r2_sr,
        total_return_pct=r2_ret,
        max_dd=r2_dd,
        win_rate=r2_wr,
        trades=r2_tr,
        passed=m2_pass,
        notes="Extended out-of-sample portfolio drift",
    )

    # M3: Cross-Asset Volatility Shock (2.5x dispersion across all 30 assets)
    r_shock = r_ho * 2.5
    v_shock = v_ho * 2.5
    r3_ret, r3_sr, _, _, r3_tr, _, r3_dd, r3_wr = simulate_portfolio_fn(
        s_ho, r_shock, v_shock, mode=mode, sizing=sizing, threshold=threshold, initial_capital=initial_capital
    )
    m3_pass = bool(m2_pass and (r3_dd <= 45.0))
    if m2_pass and not m3_pass:
        failures.append(f"M3_30_asset_volatility_shock_drawdown_{r3_dd:.1f}pct")
    rungs["M3"] = RungEvaluation(
        rung=MarketRung.M3_VOLATILITY_SHOCK,
        sharpe=r3_sr,
        total_return_pct=r3_ret,
        max_dd=r3_dd,
        win_rate=r3_wr,
        trades=r3_tr,
        passed=m3_pass,
        notes="2.5x cross-asset volatility surge",
    )

    # M4: Cross-Asset Liquidity Drought (5x fees / slippage)
    r4_ret, r4_sr, _, _, r4_tr, _, r4_dd, r4_wr = simulate_portfolio_fn(
        s_ho, r_ho, v_ho, mode=mode, sizing=sizing, threshold=threshold, fee_bps=base_fee_bps * 5.0, initial_capital=initial_capital
    )
    m4_pass = bool(m2_pass and (r4_dd <= 45.0 and r4_ret >= -35.0))
    if m2_pass and not m4_pass:
        failures.append(f"M4_30_asset_liquidity_drought_collapse_{r4_ret:.1f}pct")
    rungs["M4"] = RungEvaluation(
        rung=MarketRung.M4_LIQUIDITY_DROUGHT,
        sharpe=r4_sr,
        total_return_pct=r4_ret,
        max_dd=r4_dd,
        win_rate=r4_wr,
        trades=r4_tr,
        passed=m4_pass,
        notes="5x cross-asset turnover friction surge",
    )

    # M5: Crisis Correlation Coupling (Crypto and Equity correlation jumps to +0.95)
    # Synthesize coupled returns: individual asset returns collapse into the market factor
    market_factor = np.mean(r_ho, axis=1, keepdims=True)
    r_coupled = 0.85 * market_factor + 0.15 * r_ho  # Strong co-movement eliminate idiosyncratic hedges
    r5_ret, r5_sr, _, _, r5_tr, _, r5_dd, r5_wr = simulate_portfolio_fn(
        s_ho, r_coupled, v_ho, mode=mode, sizing=sizing, threshold=threshold, initial_capital=initial_capital
    )
    m5_pass = bool(m3_pass and m4_pass and (r5_dd <= 45.0))
    if (m3_pass and m4_pass) and not m5_pass:
        failures.append(f"M5_crisis_correlation_coupling_failure_{r5_dd:.1f}pct")
    rungs["M5"] = RungEvaluation(
        rung=MarketRung.M5_STRUCTURAL_CRASH,
        sharpe=r5_sr,
        total_return_pct=r5_ret,
        max_dd=r5_dd,
        win_rate=r5_wr,
        trades=r5_tr,
        passed=m5_pass,
        notes="Crisis coupling: correlation breakdown across all 30 assets",
    )

    # Determine highest rung passed
    highest = MarketRung.M0_IN_SAMPLE
    order = [
        MarketRung.M0_IN_SAMPLE,
        MarketRung.M1_FORWARD_HOLDOUT,
        MarketRung.M2_EXTENDED_DRIFT,
        MarketRung.M3_VOLATILITY_SHOCK,
        MarketRung.M4_LIQUIDITY_DROUGHT,
        MarketRung.M5_STRUCTURAL_CRASH,
    ]
    for r in order:
        r_key = r.value.split("_")[0].upper()
        if rungs.get(r_key, RungEvaluation(r, 0, 0, 0, 0, 0, False)).passed:
            highest = r
        else:
            break

    extrap_safe = (highest == MarketRung.M5_STRUCTURAL_CRASH)

    empirical_passed = bool(
        rungs.get("M0", RungEvaluation(MarketRung.M0_IN_SAMPLE, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M1", RungEvaluation(MarketRung.M1_FORWARD_HOLDOUT, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M2", RungEvaluation(MarketRung.M2_EXTENDED_DRIFT, 0, 0, 0, 0, 0, False)).passed
    )
    counterfactual_passed = bool(
        rungs.get("M3", RungEvaluation(MarketRung.M3_VOLATILITY_SHOCK, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M4", RungEvaluation(MarketRung.M4_LIQUIDITY_DROUGHT, 0, 0, 0, 0, 0, False)).passed
        and rungs.get("M5", RungEvaluation(MarketRung.M5_STRUCTURAL_CRASH, 0, 0, 0, 0, 0, False)).passed
    )

    return MarketExtrapolationResult(
        candidate_id=candidate_id,
        rungs=rungs,
        highest_rung_passed=highest,
        extrapolative_safe=extrap_safe,
        failure_reasons=failures,
        vol_shock_survival_ratio=r3_dd / max(r1_dd, 1.0) if r1_dd > 0 else 1.0,
        liquidity_drag_ratio=abs(r4_ret) / max(abs(r1_ret), 1.0) if abs(r1_ret) > 0 else 1.0,
        structural_retention_ratio=1.0 if extrap_safe else 0.5,
        empirical_transfer_passed=empirical_passed,
        counterfactual_stress_passed=counterfactual_passed,
    )
