"""Tests for OpenAI Mathematics Advanced Hypotheses Module.

Validates:
  - Gate -1 Causal Target Certification blocks contemporaneous lookahead leakage
  - Phase 1 Controlled 3-arm machinery verification (Signal promotes, Noise abstains, Leaked aborts)
  - The 5 advanced hypotheses: Liénard, SDE, Coupled Network, Entropy-Rate, Fourier Spectral
  - Strict boundary truncation and asset-level cross-sectional excess return panel
"""
import numpy as np
import pandas as pd

from apedp_dyn.sandbox.openai_math_advanced_hypotheses import (
    ADVANCED_HYPOTHESIS_SPECS,
    build_advanced_hypotheses_panel,
    run_preflight_controlled_verification,
    run_single_advanced_cell,
    verify_gate_minus_1_causality,
)


def test_vol_scaled_ofi_survives_volatility_warmup():
    rng = np.random.default_rng(42)
    n = 100
    universe = {}
    for symbol, scale in (("A", 0.01), ("B", 0.02)):
        returns = rng.normal(0.0, scale, n)
        universe[symbol] = pd.DataFrame({
            "r": returns,
            "price": 100.0 * np.exp(np.cumsum(returns)),
            "ofi": rng.normal(size=n),
        })

    features, _, _ = build_advanced_hypotheses_panel(universe, ("A", "B"), 15)
    scaled_ofi = features["vol_scaled_ofi"]

    assert np.isfinite(scaled_ofi).all()
    assert np.any(np.abs(scaled_ofi[-40:]) > 0.0)


def test_gate_minus_1_causality_detection():
    """Verify that Gate -1 cleanly detects contemporaneous target leakage."""
    n = 500
    rng = np.random.default_rng(42)
    contemp_r = rng.normal(0, 1, n)

    # Clean causal target: independent of contemp_r
    clean_target = rng.normal(0, 1, n)
    res_clean = verify_gate_minus_1_causality(clean_target, contemp_r, max_threshold=0.10)
    assert res_clean.passed is True
    assert res_clean.leakage_detected is False

    # Leaked target: contaminated with contemp_r
    leaked_target = clean_target + 0.5 * contemp_r
    res_leaked = verify_gate_minus_1_causality(leaked_target, contemp_r, max_threshold=0.10)
    assert res_leaked.passed is False
    assert res_leaked.leakage_detected is True
    assert abs(res_leaked.contemp_corr) >= 0.10


def test_phase_1_preflight_controlled_verification():
    """Verify that APEDP-X machinery correctly distinguishes true signal, noise, and leakage."""
    preflight = run_preflight_controlled_verification()
    assert preflight["passed_all_controls"] is True
    assert preflight["arm_a_signal"]["gated_verdict"] == "PROMOTED_BIDIRECTIONAL"
    assert preflight["arm_b_noise"]["gated_verdict"] == "TERMINAL_ABSTAIN"
    assert preflight["arm_c_leakage"]["gated_verdict"] == "FATAL_ABORT_DATA_LEAKAGE"


def test_single_cell_evaluation_15_assets_30m():
    """Verify advanced hypothesis evaluation on 15 assets at 30m horizon."""
    res = run_single_advanced_cell(
        universe_size=15,
        horizon_minutes=30,
        month_start_idx=80,
    )
    assert res["universe_size"] == 15
    assert res["horizon_minutes"] == 30
    assert len(res["evaluations"]) == 5
    assert res["cang_passed"] is True
    assert res["winning_hyp_id"] in ADVANCED_HYPOTHESIS_SPECS


def test_all_5_advanced_hypotheses_integrity():
    """Verify specifications for all 5 advanced hypotheses."""
    assert len(ADVANCED_HYPOTHESIS_SPECS) == 5
    for _hyp_id, spec in ADVANCED_HYPOTHESIS_SPECS.items():
        assert "name" in spec
        assert "scientific_origin" in spec
        assert "formula" in spec
        assert "feature_keys" in spec
        assert len(spec["feature_keys"]) >= 3
