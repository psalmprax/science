"""Unit tests for Multi-Asset Trading Policy Grammar Space across 15, 30, 60, 90 Universes.

Verifies:
  1. Exact registration of all 12 grammar symbols, grammar types, and roles.
  2. Provenance of Shared Predictor Equation A with authoritative weights from PINNED_30_CHROMOSOME.
  3. Correct stateful behavior of HYSTERESIS with PREV_POSITION, THETA_IN, THETA_OUT.
  4. Execution of the 3-strategy policy audit (Baseline, Hysteresis, Exposure-Matched) across 15, 30, 60, 90 universes.
  5. Adjudication of the three equivalences: Predictor (True), Policy (False), Economic (False).
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.sandbox.grammar_space import (
    TRADING_POLICY_SYMBOLS,
    EQUATION_A_AUTHORITATIVE_WEIGHTS,
    EQUATION_A_ROUNDED_WEIGHTS,
    TradingGrammarSymbol,
    evaluate_equation_a_score,
    op_sign,
    op_hysteresis,
    op_exposure_scale,
    op_turnover_cost,
    audit_policy_equivalences,
)
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    PINNED_30_CHROMOSOME,
    run_multi_asset_policy_grammar_audit,
)


def test_trading_policy_grammar_symbols_registration():
    """Verify all 18 symbols (12 core + 6 Candidates C/D/E) are correctly registered."""
    expected_registry = {
        "H_BB": ("Feature primitive", "Predictor input"),
        "H_OBV_MOM": ("Feature primitive", "Predictor input"),
        "NL_TAKER_MOM": ("Feature primitive", "Predictor input"),
        "NL_MASTER_SNR": ("Feature primitive", "Predictor input"),
        "S": ("Score expression", "Shared predictor output"),
        "SIGN": ("Stateless decision operator", "Baseline direction"),
        "HYSTERESIS": ("Stateful decision operator", "Entry, persistence and exit"),
        "THETA_IN": ("Threshold parameter", "Hysteresis entry boundary"),
        "THETA_OUT": ("Threshold parameter", "Hysteresis exit boundary"),
        "PREV_POSITION": ("State variable", "Previous position"),
        "EXPOSURE_SCALE": ("Scalar parameter", "Exposure matching"),
        "TURNOVER_COST": ("Execution operator", "Cost based on position changes"),
        "DEADBAND": ("Stateless decision operator", "Weak-signal filtering"),
        "TAU": ("Threshold parameter", "Deadband filtering boundary"),
        "TURNOVER_PENALTY": ("Stateful decision operator", "Cost-aware utility policy"),
        "LAMBDA": ("Penalty scalar parameter", "Turnover penalty weight"),
        "PRODUCT_H_BB_TAKER": ("Interaction operator", "Nonlinear interaction predictor term"),
        "BETA_5": ("Interaction coefficient parameter", "Nonlinear interaction weight"),
    }

    assert len(TRADING_POLICY_SYMBOLS) == 18
    for sym, (exp_type, exp_role) in expected_registry.items():
        assert sym in TRADING_POLICY_SYMBOLS, f"Symbol {sym} missing from grammar registry"
        item = TRADING_POLICY_SYMBOLS[sym]
        assert isinstance(item, TradingGrammarSymbol)
        assert item.grammar_type == exp_type, f"{sym} type mismatch: {item.grammar_type} != {exp_type}"
        assert item.role == exp_role, f"{sym} role mismatch: {item.role} != {exp_role}"


def test_equation_a_authoritative_weights_provenance():
    """Verify Equation A weights match PINNED_30_CHROMOSOME in multi_asset_signals_sandbox.py."""
    # Active indices in PINNED_30_CHROMOSOME: [9, 16, 24, 25]
    w_30 = PINNED_30_CHROMOSOME.weights
    assert np.isclose(EQUATION_A_AUTHORITATIVE_WEIGHTS["H_BB"], w_30[9], atol=1e-6)
    assert np.isclose(EQUATION_A_AUTHORITATIVE_WEIGHTS["H_OBV_MOM"], w_30[16], atol=1e-6)
    assert np.isclose(EQUATION_A_AUTHORITATIVE_WEIGHTS["NL_TAKER_MOM"], w_30[24], atol=1e-6)
    assert np.isclose(EQUATION_A_AUTHORITATIVE_WEIGHTS["NL_MASTER_SNR"], w_30[25], atol=1e-6)

    # Rounded weights match specification
    assert EQUATION_A_ROUNDED_WEIGHTS["H_BB"] == 0.8111
    assert EQUATION_A_ROUNDED_WEIGHTS["H_OBV_MOM"] == -1.4272
    assert EQUATION_A_ROUNDED_WEIGHTS["NL_TAKER_MOM"] == 0.5984
    assert EQUATION_A_ROUNDED_WEIGHTS["NL_MASTER_SNR"] == 0.6740


def test_stateful_hysteresis_operator():
    """Verify HYSTERESIS decision operator path dependence on PREV_POSITION."""
    # Sequence of scores: 0.10 (below theta_in 0.1979), 0.25 (above), 0.10 (between), -0.05 (below theta_out 0.0)
    scores = np.array([0.10, 0.25, 0.10, -0.05])
    pos = op_hysteresis(scores, theta_in=0.1979, theta_out=0.0)

    # Step 0: score=0.10, prev=0 -> 0.0 (below theta_in)
    assert pos[0] == 0.0
    # Step 1: score=0.25, prev=0 -> +1.0 (exceeds theta_in)
    assert pos[1] == 1.0
    # Step 2: score=0.10, prev=+1 -> +1.0 (stateful persistence: above theta_out 0.0)
    assert pos[2] == 1.0
    # Step 3: score=-0.05, prev=+1 -> 0.0 (drops below theta_out 0.0 -> exit)
    assert pos[3] == 0.0


def test_exposure_scale_and_turnover_cost():
    """Verify EXPOSURE_SCALE and TURNOVER_COST operators."""
    pos_base = np.array([0.0, 1.0, -1.0, 0.0])
    scaled = op_exposure_scale(pos_base, alpha=1.25)
    np.testing.assert_allclose(scaled, np.array([0.0, 1.25, -1.25, 0.0]))

    cost = op_turnover_cost(pos_curr=np.array([1.0]), pos_prev=np.array([0.0]), capital=100000.0, fee_bps=5.0)
    # Turnover = 1.0, fee = 5 bps = 0.0005, cost = 100000 * 1 * 0.0005 = 50.0
    assert np.isclose(cost, 50.0)


def test_candidates_c_d_e_operators_and_catalog():
    """Verify Candidate C (Deadband), D (Turnover Penalty), E (Interaction), and Catalog."""
    from apedp_dyn.sandbox.grammar_space import (
        GRAMMAR_CANDIDATE_CATALOG,
        op_deadband,
        op_turnover_penalty,
        op_product_interaction,
        evaluate_equation_e_score,
    )

    # 1. Verify candidate catalog has A, B, H, C, D, E
    assert set(GRAMMAR_CANDIDATE_CATALOG.keys()) == {"A", "B", "H", "C", "D", "E"}
    assert GRAMMAR_CANDIDATE_CATALOG["C"].name == "Deadband"
    assert GRAMMAR_CANDIDATE_CATALOG["D"].name == "Turnover penalty"
    assert GRAMMAR_CANDIDATE_CATALOG["E"].name == "Interaction predictor"

    # 2. Test Candidate C (Stateless Deadband)
    scores = np.array([0.25, 0.10, -0.05, -0.25])
    pos_c = op_deadband(scores, tau=0.1979)
    # Step 0: 0.25 >= 0.1979 -> +1.0
    # Step 1: 0.10 in (-0.1979, 0.1979) -> 0.0 (stateless! no memory of +1.0)
    # Step 2: -0.05 in (-0.1979, 0.1979) -> 0.0
    # Step 3: -0.25 <= -0.1979 -> -1.0
    np.testing.assert_array_equal(pos_c, np.array([1.0, 0.0, 0.0, -1.0]))

    # 3. Test Candidate D (Turnover Penalty Utility)
    # With prev=1.0 and score=0.10, utility for p=1 is 0.10 - 0.20*0 = 0.10.
    # Utility for p=0 is 0.0 - 0.20*1 = -0.20. Best is p=1.0 (inertia due to penalty).
    pos_d = op_turnover_penalty(scores=np.array([0.25, 0.10]), lambda_pen=0.20, init_pos=np.array([0.0]))
    assert pos_d[0] == 1.0
    assert pos_d[1] == 1.0  # Kept position because switching to 0 incurs penalty -0.20

    # 4. Test Candidate E (Product Interaction)
    f1 = np.array([2.0, -1.0])
    f2 = np.array([3.0, 4.0])
    prod = op_product_interaction(f1, f2)
    np.testing.assert_array_equal(prod, np.array([6.0, -4.0]))


def test_multi_asset_policy_grammar_audit_all_universes():
    """Verify full audit runs across 15, 30, 60, and 90 asset universes with correct equivalences."""
    res = run_multi_asset_policy_grammar_audit(
        universes=(15, 30, 60, 90),
        holdout_only=True,
        use_pinned=True,
        verbose=False,
    )

    for n_str in ("15", "30", "60", "90"):
        assert n_str in res["universes"]
        u_data = res["universes"][n_str]
        assert "candidate_A" in u_data
        assert "candidate_H" in u_data
        assert "candidate_C" in u_data
        assert "candidate_D" in u_data
        assert "candidate_E" in u_data
        eqs = u_data["equivalences_A_vs_H"]

        # 1. Predictor equivalence must be TRUE (same underlying equation)
        assert eqs["predictor_equivalence"] is True

        # 2. Policy equivalence must be FALSE (hysteresis delays exits, changing positions)
        assert eqs["policy_equivalence"] is False
        assert eqs["policy_divergence_pct"] > 5.0  # Divergence around ~6.7-7.0%

        # 3. Economic equivalence must be FALSE (net returns and Sharpe differ significantly)
        assert eqs["economic_equivalence"] is False
        assert eqs["delta_net_ret"] > 1.0  # Divergence in net return > 1.0 pp
