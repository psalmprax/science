"""Unit tests for Depth-2 Non-Linear Genetic Algorithm Equation Discovery.

Verifies:
  - Feature catalog integrity (Depth-1 primitives vs Depth-2 non-linear operations)
  - Asset-level panel generation with strict boundary truncation
  - Gate -1 causal target verification and Frisch-Waugh-Lovell purging
  - Genetic Algorithm evolution dynamics (selection of Depth-2 terms, term bounds, condition number)
  - Full cell execution and epistemic verdict adjudication
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.sandbox.ga_depth2_equation_discovery import (
    DEPTH2_FEATURE_CATALOG,
    DEPTH2_HORIZONS,
    DEPTH2_KEYS,
    DEPTH2_UNIVERSES,
    build_depth2_panel,
    run_depth2_genetic_algorithm,
    run_single_depth2_ga_cell,
    verify_gate_minus_1_causality,
)
from apedp_dyn.sandbox.multi_asset_signals_sandbox import load_universe


def test_depth2_feature_catalog_integrity():
    """Verify that all features in DEPTH2_FEATURE_CATALOG are well-defined."""
    assert len(DEPTH2_FEATURE_CATALOG) == 24
    d1_count = sum(1 for e in DEPTH2_FEATURE_CATALOG if e["depth"] == "1")
    d2_count = sum(1 for e in DEPTH2_FEATURE_CATALOG if e["depth"] == "2")
    assert d1_count == 8
    assert d2_count == 16
    assert len(DEPTH2_KEYS) == 24


def test_depth2_panel_construction_and_causality():
    """Verify strictly causal panel construction on 15 crypto assets."""
    universe, symbols = load_universe("15")
    X_dict, y_flat, r_contemp_flat = build_depth2_panel(universe, symbols, horizon_minutes=30)

    # Check that all keys are present and have equal length
    assert len(X_dict) == 24
    n_pts = len(y_flat)
    assert n_pts > 0
    for k, v in X_dict.items():
        assert len(v) == n_pts
        assert np.all(np.isfinite(v))

    # Gate -1 Causal Check
    passed, r_leak = verify_gate_minus_1_causality(y_flat, r_contemp_flat, max_threshold=0.10)
    assert abs(r_leak) < 0.10
    assert passed is True


def test_ga_evolution_discovers_depth2_terms():
    """Verify that the Genetic Algorithm evolves valid equations containing Depth-2 terms."""
    universe, symbols = load_universe("15")
    X_dict, y_flat, r_contemp_flat = build_depth2_panel(universe, symbols, horizon_minutes=30)

    N = len(symbols)
    tr_s = slice(80 * N, (80 + 21) * N)
    X_tr = {k: v[tr_s] for k, v in X_dict.items()}
    y_tr = y_flat[tr_s]

    best_chrom = run_depth2_genetic_algorithm(
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

    # Verify at least one Depth-2 term was discovered and selected by the GA
    catalog_map = {entry["key"]: entry["depth"] for entry in DEPTH2_FEATURE_CATALOG}
    depths = [catalog_map[t] for t in best_chrom.terms]
    assert "2" in depths


def test_single_cell_execution():
    """Verify single cell execution on 15 assets at 15m horizon."""
    cell = run_single_depth2_ga_cell(
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
    assert cell.gated_verdict in (
        "PROMOTED_BIDIRECTIONAL",
        "PROMOTED_FORWARD_ONLY",
        "REFUSED_HOLDOUT_GATE_4",
        "TERMINAL_ABSTAIN",
    )
