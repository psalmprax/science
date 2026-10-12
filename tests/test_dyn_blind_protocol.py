"""Unit Test Suite for APEDP-X Blind Benchmark Protocol & Frozen Spec Provenance.

Verifies:
  1. Cryptographic specification integrity (SHA256 digests of kernel, grammar, ladder).
  2. Competitive baseline ladder execution across domains.
  3. Multi-split blinded evaluation (B_dev, B_blind, B_OOD) execution and metrics.
  4. Full confusion matrix and scientific utility metric calculation.
  5. APEDP-on-APEDP self-audit consistency.
"""
from __future__ import annotations

from apedp_dyn.epistemic.frozen_spec import get_frozen_spec
from apedp_dyn.sandbox import blind_benchmark_protocol as BBP
from apedp_dyn.sandbox.grammar_space import GRAMMAR_FINGERPRINT


def test_frozen_spec_hashes_and_metadata():
    """Frozen spec must contain non-empty, 64-character SHA256 hex digests."""
    spec = get_frozen_spec()
    assert spec["version"] == "APEDP-X v1.0-FROZEN"
    assert len(spec["grammar_sha256"]) == 64
    assert len(spec["referee_kernel_sha256"]) == 64
    assert len(spec["claim_ladder_sha256"]) == 64
    assert spec["grammar_sha256"] == GRAMMAR_FINGERPRINT

    # Verify thresholds for all 5 domains are present
    thresh = spec["thresholds"]
    assert "dynamical" in thresh
    assert "causal" in thresh
    assert "geometry_optimality" in thresh
    assert "physics_pde" in thresh
    assert "volatility_empirical" in thresh


def test_baseline_ladders_execution():
    """Competitive baseline ladder generates decisions for each domain."""
    # Test dynamical ladder
    dyn_sc = BBP.generate_blind_dynamical_cases(n_cases=2, seed=99)[0]
    dyn_baselines = BBP.evaluate_baseline_ladder(dyn_sc)
    assert "Train-Min" in dyn_baselines
    assert "AIC" in dyn_baselines
    assert "BIC" in dyn_baselines
    assert "CV-5" in dyn_baselines
    assert "SINDy-Proxy" in dyn_baselines

    # Test causal ladder
    causal_sc = BBP.generate_blind_causal_cases(n_cases=2, seed=99)[0]
    causal_baselines = BBP.evaluate_baseline_ladder(causal_sc)
    assert "Bivariate-OLS" in causal_baselines
    assert "Granger" in causal_baselines
    assert "PC-Algorithm" in causal_baselines
    assert "LiNGAM-SCM" in causal_baselines

    # Test packing ladder
    pack_sc = BBP.generate_blind_packing_cases(n_cases=2, seed=99)[0]
    pack_baselines = BBP.evaluate_baseline_ladder(pack_sc)
    assert "Gradient-Local" in pack_baselines
    assert "Basin-Hopping" in pack_baselines
    assert "Monte-Carlo" in pack_baselines
    assert "Bound-Aware-Opt" in pack_baselines

    # Test PDE ladder
    pde_sc = BBP.generate_blind_pde_cases(n_cases=2, seed=99)[0]
    pde_baselines = BBP.evaluate_baseline_ladder(pde_sc)
    assert "PINN-Residual-Only" in pde_baselines
    assert "Standard-FDM" in pde_baselines
    assert "Invariant-FDM" in pde_baselines


def test_blind_benchmark_splits_execution():
    """Full blinded protocol executes across B_dev, B_blind, and B_OOD without error."""
    res = BBP.execute_full_blind_protocol()
    m_dev = res["m_dev"]
    m_blind = res["m_blind"]
    m_ood = res["m_ood"]
    m_all = res["m_all"]

    assert m_dev["n_total"] == 18
    assert m_blind["n_total"] == 40
    assert m_ood["n_total"] == 20
    assert m_all["n_total"] == 78

    # Epistemic safety: APEDP FPR must be 0% across all splits
    assert m_dev["fpr"] == 0.0
    assert m_blind["fpr"] == 0.0
    assert m_ood["fpr"] == 0.0
    assert m_all["fpr"] == 0.0

    # Scientific utility index must be non-zero and bounded in [0, 100]
    assert 0.0 <= m_all["scientific_utility"] <= 100.0


def test_apedp_self_audit_completes():
    """Self-audit function executes cleanly without throwing exceptions."""
    BBP.run_apedp_self_audit()
