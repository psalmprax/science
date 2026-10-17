"""Unit tests for Depth-3 Non-Linear Genetic Algorithm Equation Discovery.

Verifies:
  - Feature catalog integrity (8 Depth-1 primitives, 8 Depth-2 operations, 16 Depth-3 3-way operations)
  - Asset-level panel generation with strict boundary truncation
  - Gate -1 causal target verification and Frisch-Waugh-Lovell purging
  - Genetic Algorithm evolution dynamics (selection of Depth-3 terms, term bounds, condition number)
  - Full cell execution and epistemic verdict adjudication
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.sandbox.ga_depth3_equation_discovery import (
    DEPTH3_FEATURE_CATALOG,
    DEPTH3_KEYS,
    build_depth3_panel,
    run_depth3_genetic_algorithm,
    run_single_depth3_ga_cell,
    verify_gate_minus_1_causality,
)
from apedp_dyn.sandbox.multi_asset_signals_sandbox import load_universe


def test_depth3_feature_catalog_integrity():
    """Verify that all features in DEPTH3_FEATURE_CATALOG are well-defined."""
    assert len(DEPTH3_FEATURE_CATALOG) == 32
    d1_count = sum(1 for e in DEPTH3_FEATURE_CATALOG if e["depth"] == "1")
    d2_count = sum(1 for e in DEPTH3_FEATURE_CATALOG if e["depth"] == "2")
    d3_count = sum(1 for e in DEPTH3_FEATURE_CATALOG if e["depth"] == "3")
    assert d1_count == 8
    assert d2_count == 8
    assert d3_count == 16
    assert len(DEPTH3_KEYS) == 32


def test_depth3_panel_construction_and_causality():
    """Verify strictly causal panel construction on 15 crypto assets."""
    universe, symbols = load_universe("15")
    X_dict, y_flat, r_contemp_flat = build_depth3_panel(universe, symbols, horizon_minutes=30)

    assert len(X_dict) == 32
    n_pts = len(y_flat)
    assert n_pts > 0
    for _k, v in X_dict.items():
        assert len(v) == n_pts
        assert np.all(np.isfinite(v))

    # Gate -1 Causal Check
    passed, r_leak = verify_gate_minus_1_causality(y_flat, r_contemp_flat, max_threshold=0.10)
    assert abs(r_leak) < 0.10
    assert passed is True


def test_ga_evolution_discovers_depth3_terms():
    """Verify that the Genetic Algorithm evolves valid equations containing Depth-3 terms."""
    universe, symbols = load_universe("15")
    X_dict, y_flat, _r_contemp_flat = build_depth3_panel(universe, symbols, horizon_minutes=30)

    N = len(symbols)
    tr_s = slice(80 * N, (80 + 21) * N)
    X_tr = {k: v[tr_s] for k, v in X_dict.items()}
    y_tr = y_flat[tr_s]

    best_chrom = run_depth3_genetic_algorithm(
        X_dict=X_tr,
        y_train=y_tr,
        pop_size=40,
        generations=15,
        min_terms=2,
        max_terms=5,
        seed=123,
    )

    assert len(best_chrom.terms) >= 2
    assert len(best_chrom.terms) <= 5
    assert best_chrom.condition_number <= 1000.0
    assert np.isfinite(best_chrom.train_r2)
    assert len(best_chrom.fitted_formula) > 0

    catalog_map = {entry["key"]: entry["depth"] for entry in DEPTH3_FEATURE_CATALOG}
    depths = [catalog_map[t] for t in best_chrom.terms]
    assert "3" in depths


def test_single_cell_execution_depth3():
    """Verify single cell execution on 15 assets at 15m horizon."""
    cell = run_single_depth3_ga_cell(
        universe_size=15,
        horizon_minutes=15,
        month_start_idx=80,
        month_len=21,
        pop_size=30,
        generations=10,
        seed=99,
    )

    assert cell.cell_id == "U15_H15m"
    assert cell.universe_size == 15
    assert cell.horizon_minutes == 15
    assert cell.condition_number <= 1000.0
    assert cell.gate_minus_1_passed is True
    assert cell.best_fit_verdict.startswith("PROMOTES")
    assert cell.depth_breakdown["depth_3"] >= 1
    assert cell.gated_verdict in (
        "PROMOTED_BIDIRECTIONAL",
        "PROMOTED_FORWARD_ONLY",
        "REFUSED_HOLDOUT_GATE_4",
        "TERMINAL_ABSTAIN",
    )
