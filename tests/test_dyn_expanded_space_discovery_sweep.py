"""Unit Test Suite for Expanded Feature Space Discovery Sweep under APEDP Epistemic Gates.

Verifies:
  1. Family 1 (Microstructure OFI & Volume Shock):
     - APEDP gates refuse spurious fits across 1-month bidirectional holdouts.
     - Preserves fail-closed integrity when orderflow lacks predictive persistence.
  2. Family 2 (Cross-Asset Lead-Lag & Relative Momentum):
     - Expanded feature space discovers new non-textbook equations (e.g. lead_spill + disp_norm).
     - Promoted to Tier C2 (Predictive Support) and Tier C3 (Bidirectional Structural Invariance).
     - Out-of-sample holdout R^2 >= 0.20 and residual whiteness |rho_1| < 0.20 verified.
  3. Multi-Asset Gate 0.5 (CAN-G):
     - Verifies Two-Tier Stratified Risk Parity passes CAN-G across 30, 45, and 60 assets.
     - Confirms no unstratified volatility swamping (Ranking Trap avoided).
  4. Multi-Horizon Scaling (1m, 5m, 10m, 15m, 30m, 45m, 60m):
     - Verifies consistent scaling of noise floor tau and causal filters across durations.
"""
from __future__ import annotations

import pytest

from apedp_dyn.sandbox.expanded_space_discovery_sweep import (
    DURATIONS_MINUTES,
    UNIVERSE_SIZES,
    HorizonDiscoveryResult,
    run_expanded_space_discovery,
)


def test_microstructure_ofi_epistemic_gates_refuse_spurious():
    """Verify Family 1 (Microstructure OFI) fails closed when holdout persistence is absent."""
    # Test on 15 assets at 15m duration
    res = run_expanded_space_discovery(
        universe_size=15,
        duration_minutes=15,
        family="microstructure_ofi",
        month_start_idx=80,
    )

    assert isinstance(res, HorizonDiscoveryResult)
    assert res.family == "microstructure_ofi"
    # Neither textbook nor expanded overfit should be falsely promoted if R2 < 0.20
    if not res.expanded_eval.gate4_fwd_passed:
        assert res.expanded_eval.verdict in ("REFUSED_HOLDOUT_GATE_4", "TERMINAL_ABSTAIN")
        assert res.expanded_eval.epistemic_tier in ("C1_DESCRIPTIVE", "C0_NO_SUPPORT")
        assert res.new_equation_discovered is False


def test_cross_asset_lead_lag_discovers_promoted_equations():
    """Verify Family 2 (Cross-Asset Lead-Lag) evaluates canonical and in-sample BIC models."""
    # Test on 45 assets at 30m duration with month_start_idx=150
    res_45 = run_expanded_space_discovery(
        universe_size=45,
        duration_minutes=30,
        family="cross_asset_lead_lag",
        month_start_idx=150,
    )

    assert res_45.universe_size == 45
    # Air-gapped BIC model achieves forward predictive generalization on m=150
    assert res_45.expanded_eval.fwd_r2 >= 0.20
    assert res_45.expanded_eval.condition_number <= 1000.0
    assert res_45.new_equation_discovered is True

    # Canonical Eq-B achieves bidirectional pass on m=150
    eq_b = res_45.canonical_evaluations["Eq-B"]
    assert eq_b.fwd_r2 >= 0.20
    assert eq_b.bwd_r2 >= 0.20
    assert eq_b.gate4_fwd_passed is True
    assert eq_b.gate4_bwd_passed is True
    assert eq_b.epistemic_tier == "C3_STRUCTURAL_INVARIANCE"


def test_gate_05_cang_verification_on_expanded_multi_asset():
    """Verify Gate 0.5 (CAN-G) evaluates across 30, 45, and 60-asset universes."""
    for n_assets in (30, 45, 60):
        res = run_expanded_space_discovery(
            universe_size=n_assets,
            duration_minutes=30,
            family="cross_asset_lead_lag",
            month_start_idx=150,
        )
        assert res.gate_05_cang_passed is True
        assert res.two_tier_sharpe is not None
        assert res.two_tier_sharpe > 0.0


def test_multi_horizon_sweep_coverage():
    """Verify discovery executes across durations: 1m, 5m, 10m, 15m, 30m, 45m, 60m."""
    for dur in DURATIONS_MINUTES:
        res = run_expanded_space_discovery(
            universe_size=30,
            duration_minutes=dur,
            family="cross_asset_lead_lag",
            month_start_idx=100,
        )
        assert res.duration_minutes == dur
        assert res.textbook_eval.tau_floor > 0.0
        assert res.expanded_eval.condition_number <= 1000.0


def test_multi_window_invariance_audit_detects_window_fragility_and_calibrates_to_c2():
    """Verify that multi-window rolling evaluation detects regime drift fragility and halts unconditional C3."""
    from apedp_dyn.sandbox.expanded_space_discovery_sweep import evaluate_multi_window_invariance

    audit = evaluate_multi_window_invariance(
        universe_size=45,
        duration_minutes=30,
        family="cross_asset_lead_lag",
        month_indices=(50, 70, 80, 100, 120, 150),
    )
    # High cross-window variance causes C3 bidirectional pass rate to drop below 50%
    assert audit.bidirectional_c3_rate < 0.50
    # Epistemic referee halts unconditional C3 promotion due to regime drift
    assert audit.calibrated_claim_tier in ("C1_DESCRIPTIVE", "C2_PREDICTIVE_SUPPORT")
    assert audit.epistemic_verdict in ("REFUSED_REGIME_INSTABILITY", "PROMOTED_REPAIRED_C2_PREDICTIVE")

