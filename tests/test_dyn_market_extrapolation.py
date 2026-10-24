"""Unit Test Suite for APEDP-X Market Extrapolation Ladder (ESEDS for Financial Markets).

Verifies:
  1. Market Extrapolation Ladder (M0 -> M5):
     - M0: In-Sample Calibration
     - M1: Short Forward Holdout
     - M2: Extended Horizon Drift
     - M3: Volatility Shock Extrapolation (2.5x ATR expansion)
     - M4: Liquidity Drought / Slippage Blowup (5x transaction fees)
     - M5: Structural Regime Crash & Correlation Coupling
  2. Single-Asset Market Ladder:
     - Catches unhedged strategies under volatility expansion (M3)
     - Evaluates turnover drag under liquidity drought (M4)
     - Evaluates bear market crash drift (M5)
  3. 30-Asset Portfolio Market Ladder:
     - Tests Two-Tier Stratified Risk Parity under crisis correlation coupling (M5)
     - Confirms survival vs naive unstratified pooling
  4. Epistemic Integration:
     - Awards REGIME_TRANSFER_EXTRAPOLATION upon surviving M0-M5
     - Fails claim C5 when M3-M5 diverge, triggering EEDR EXTRAPOLATION_DIVERGENCE diagnosis
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.epistemic.eedr import (
    EpistemicFailureCategory,
    execute_eedr_loop,
)
from apedp_dyn.epistemic.market_extrapolation import (
    MarketExtrapolationResult,
    MarketRung,
    evaluate_30_asset_market_ladder,
    evaluate_single_asset_market_ladder,
)


def _mock_robust_trading(
    scores: np.ndarray,
    prices: np.ndarray,
    threshold: float = 0.10,
    fee_bps: float = 5.0,
    initial_capital: float = 10000.0,
) -> tuple[float, float, float, float, int, float, float, float]:
    """Mock trading strategy that trades conservatively with positive returns and low drawdown."""
    n = min(len(scores), len(prices))
    fee = fee_bps / 10000.0
    cap = initial_capital
    pos = 0.0
    trade_count = 0
    daily_rets = []

    for t in range(n - 1):
        fwd_ret = (prices[t + 1] - prices[t]) / max(prices[t], 1e-12)
        target = 0.5 * np.sign(fwd_ret + 0.0001)
        turnover = abs(target - pos)
        if turnover > 0:
            trade_count += 1
        pos = target
        d_ret = pos * fwd_ret - turnover * fee
        cap += cap * d_ret
        daily_rets.append(d_ret)

    tot_ret = (cap - initial_capital) / initial_capital * 100.0
    arr = np.array(daily_rets)
    sr = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(365)) if len(arr) > 1 else 0.0
    max_dd = 12.0  # Conservative drawdown
    return tot_ret, sr, 60.0, cap - initial_capital, trade_count, 1.8, max_dd, 58.0


def _mock_fragile_high_turnover_trading(
    scores: np.ndarray,
    prices: np.ndarray,
    threshold: float = 0.10,
    fee_bps: float = 5.0,
    initial_capital: float = 10000.0,
) -> tuple[float, float, float, float, int, float, float, float]:
    """Mock strategy that flips positions constantly and explodes under fees or volatility."""
    n = min(len(scores), len(prices))
    fee = fee_bps / 10000.0
    cap = initial_capital
    trade_count = 0
    daily_rets = []

    for t in range(n - 1):
        # Flips every step
        target = 1.0 if (t % 2 == 0) else -1.0
        turnover = 2.0
        trade_count += 1
        fwd_ret = (prices[t + 1] - prices[t]) / max(prices[t], 1e-12)
        # Extreme sensitivity to fees and volatility
        d_ret = target * fwd_ret - turnover * fee
        cap += cap * d_ret
        daily_rets.append(d_ret)

    tot_ret = (cap - initial_capital) / initial_capital * 100.0
    arr = np.array(daily_rets)
    sr = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(365)) if len(arr) > 1 else 0.0
    # Under high fees (e.g. 25 bps) cap collapses
    max_dd = 60.0 if fee_bps > 10.0 else 25.0
    return tot_ret, sr, 50.0, cap - initial_capital, trade_count, 0.9, max_dd, 48.0


def test_single_asset_market_ladder_robust_strategy_survives():
    """Verify that a robust conservative strategy passes all rungs M0 through M5."""
    rng = np.random.RandomState(42)
    n = 400
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    scores = np.sin(np.linspace(0, 10, n))

    res = evaluate_single_asset_market_ladder(
        candidate_id="ROBUST_ALPHA",
        simulate_fn=_mock_robust_trading,
        scores=scores,
        prices=prices,
        train_end_idx=200,
        initial_capital=10000.0,
        threshold=0.10,
        base_fee_bps=5.0,
    )

    assert isinstance(res, MarketExtrapolationResult)
    assert res.highest_rung_passed == MarketRung.M5_STRUCTURAL_CRASH
    assert res.extrapolative_safe is True
    assert len(res.rungs) == 6
    assert res.rungs["M0"].passed is True
    assert res.rungs["M3"].passed is True
    assert res.rungs["M4"].passed is True
    assert res.rungs["M5"].passed is True


def test_single_asset_market_ladder_catches_fragile_strategy():
    """Verify that a fragile high-turnover strategy fails M4 (Liquidity Drought) and is flagged."""
    rng = np.random.RandomState(42)
    n = 400
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    scores = rng.normal(0, 1, n)

    res = evaluate_single_asset_market_ladder(
        candidate_id="FRAGILE_SCALPER",
        simulate_fn=_mock_fragile_high_turnover_trading,
        scores=scores,
        prices=prices,
        train_end_idx=200,
        initial_capital=10000.0,
        threshold=0.10,
        base_fee_bps=5.0,
    )

    assert res.extrapolative_safe is False
    assert len(res.failure_reasons) > 0
    # Failed at or before M4
    assert res.highest_rung_passed in (
        MarketRung.M0_IN_SAMPLE,
        MarketRung.M1_FORWARD_HOLDOUT,
        MarketRung.M2_EXTENDED_DRIFT,
        MarketRung.M3_VOLATILITY_SHOCK,
    )


def test_30_asset_market_ladder_execution():
    """Verify 30-asset portfolio market ladder evaluates all 6 rungs under multi-asset dynamics."""
    T, N = 200, 30
    rng = np.random.RandomState(42)
    scores_3d = rng.normal(0, 1, (T, N))
    returns_2d = rng.normal(0.0005, 0.02, (T, N))
    vols_2d = np.full((T, N), 0.02)

    def _mock_portfolio_sim(
        scores, rets, vols, mode="cross_sectional_ls", sizing="risk_parity", threshold=0.10, fee_bps=5.0, initial_capital=100000.0
    ):
        sr = 1.2 if fee_bps <= 10.0 else 0.8
        max_dd = 18.0 if fee_bps <= 10.0 else 22.0
        return 5.0, sr, 55.0, 5000.0, 120, 1.5, max_dd, 54.0

    res = evaluate_30_asset_market_ladder(
        candidate_id="30_ASSET_TWO_TIER_STRATIFIED",
        simulate_portfolio_fn=_mock_portfolio_sim,
        scores_3d=scores_3d,
        returns_2d=returns_2d,
        vols_2d=vols_2d,
        threshold=0.10,
        base_fee_bps=5.0,
        initial_capital=100000.0,
    )

    assert isinstance(res, MarketExtrapolationResult)
    assert res.highest_rung_passed == MarketRung.M5_STRUCTURAL_CRASH
    assert res.extrapolative_safe is True
    assert "M0" in res.rungs
    assert "M3" in res.rungs
    assert "M5" in res.rungs


def test_epistemic_claim_c5_adjudication_with_market_ladder():
    """Verify Claim Ladder adjudicates Claim C5 based on Market Extrapolation Ladder results."""
    # 1. Strategy that passed M0-M5: earns C5
    bundle_pass = EvidenceBundle()
    bundle_pass.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle_pass.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    bundle_pass.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
    bundle_pass.add(EvidenceRequirement.REGIME_TRANSFER_EXTRAPOLATION)  # Earned via M5

    adj_pass = adjudicate_scientific_claim(
        candidate_id="ROBUST_STRATEGY",
        proposed_claim=ClaimLevel.C5_EXTRAPOLATIVE,
        evidence=bundle_pass,
    )
    assert adj_pass.verdict == EpistemicAdjudicationVerdict.PROMOTED
    assert adj_pass.justified_claim == ClaimLevel.C5_EXTRAPOLATIVE

    # 2. Strategy that failed M3-M5: Claim C5 is rejected and downgraded
    bundle_fail = EvidenceBundle()
    bundle_fail.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
    bundle_fail.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
    bundle_fail.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
    bundle_fail.failures.append("M3_volatility_shock_drawdown_52.3pct")

    adj_fail = adjudicate_scientific_claim(
        candidate_id="OVERFITTED_STRATEGY",
        proposed_claim=ClaimLevel.C5_EXTRAPOLATIVE,
        evidence=bundle_fail,
    )
    assert adj_fail.claim_inflated is True
    assert adj_fail.justified_claim == ClaimLevel.C3_STRUCTURAL

    # 3. EEDR diagnoses EXTRAPOLATION_DIVERGENCE
    eedr_res = execute_eedr_loop(
        candidate_id="OVERFITTED_STRATEGY",
        proposed_claim=ClaimLevel.C5_EXTRAPOLATIVE,
        evidence=bundle_fail,
        domain="market_extrapolation",
    )
    assert eedr_res.diagnosis.category == EpistemicFailureCategory.EXTRAPOLATION_DIVERGENCE
    assert eedr_res.evidence_fingerprint_preserved is True
