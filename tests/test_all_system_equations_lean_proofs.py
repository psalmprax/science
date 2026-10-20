"""Automated Unit Tests for All 323 System Equations Lean 4 Verification.

Verifies:
  - 315 genuine equations across 10 families compile with Lean 4 exit code 0 -> P2_VERIFIED
  - 8 refuted records fail Lean 4 proofs with sorry -> P0_NO_PROOF
    (the 2 built-in traps plus 6 grammar-space candidates whose declared
    saturation bounds are not provable over their formulas: unbounded
    degree-3 products and heaviside-indicator products that fail closed on
    AST refutation; specified in CHANGELOG "Lean-proof inventory
    reconciliation" 2026-10-19, not a threshold change)
  - Zero P1 regressions across the entire suite of 323 equations
"""
from __future__ import annotations

import json

import pytest

from apedp.mathematical.mathematical_layer import ProofClaimStatus
from apedp_dyn.epistemic.lean_universal_bridge import LeanUniversalBridge, VerificationBackend
from apedp_dyn.sandbox.build_all_system_equations_lean_proofs import (
    ARTIFACT_DIR,
    REPO_ROOT,
    assemble_exhaustive_candidate_specifications,
)

_PROVENANCE_INPUTS = (
    "fluid_ode_dynamical_sweep_provenance.json",
    "openai_math_advanced_hypotheses_sweep_provenance.json",
    "ga_depth2_sweep_provenance.json",
    "ga_depth3_sweep_provenance.json",
    "expanded_space_discovery_provenance.json",
)


def _require_provenance_inputs():
    missing = [
        name for name in _PROVENANCE_INPUTS
        if not (ARTIFACT_DIR / name).is_file() and not (REPO_ROOT / name).is_file()
    ]
    if missing:
        pytest.skip(f"exhaustive inventory requires generated provenance inputs: {missing}")


def test_system_wide_candidate_inventory():
    """Verify that all 323 candidate specifications are assembled across all 10 sources."""
    _require_provenance_inputs()
    specs = assemble_exhaustive_candidate_specifications()
    assert len(specs) == 323
    candidate_ids = {s.candidate_id for s in specs}

    # Verify presence of key families
    assert "FLUID-CELL-U15-H1M-EQ1_NAVIERSTOKES" in candidate_ids
    assert "MATH-CELL-U15-H15M-HYP1_LIENARD_OSCILLATOR" in candidate_ids
    assert "GA-D2-CHAMPION-U15_H15M" in candidate_ids
    assert "GA-D3-CHAMPION-U90_H60M" in candidate_ids
    assert "WALKFORWARD-WIN-M50" in candidate_ids
    assert "EXP-SPLIT-SPLIT_80-AIR_GAPPED_BIC_WINNER" in candidate_ids
    assert "TRADING-SCORE-N15" in candidate_ids
    assert "TRADING-SCORE-N90" in candidate_ids
    assert "STRUCT-MASTER-SIZING-PHI" in candidate_ids
    assert "GRAMMAR-D2-TANH_VELOCITY" in candidate_ids
    assert "GRAMMAR-D3-VEL_TANH_OFI_SPREAD_3W" in candidate_ids
    assert "PHYS-ENERGY-LEAK-TRAP" in candidate_ids
    assert "FIN-CUBIC-RUNAWAY-POLYNOMIAL" in candidate_ids


def test_provenance_file_integrity():
    """Verify provenance JSON records all 323 evaluations with exact compiler verdicts."""
    provenance_path = ARTIFACT_DIR / "all_system_equations_lean_provenance.json"
    if not provenance_path.is_file():
        pytest.skip("generated Lean proof provenance artifact is not available")

    with open(provenance_path) as f:
        data = json.load(f)

    assert data["total_equations_evaluated"] == 323
    assert data["p2_verified_count"] == 315
    assert data["p0_refuted_count"] == 8
    assert data["p1_candidate_count"] == 0

    results_by_id = {r["candidate_id"]: r for r in data["results"]}
    assert results_by_id["PHYS-ENERGY-LEAK-TRAP"]["proof_claim_status"] == "P0"
    assert results_by_id["FIN-CUBIC-RUNAWAY-POLYNOMIAL"]["proof_claim_status"] == "P0"
    assert results_by_id["STRUCT-MASTER-SIZING-PHI"]["proof_claim_status"] == "P2"
    assert results_by_id["GA-D3-CHAMPION-U90_H60M"]["proof_claim_status"] == "P2"
    assert results_by_id["TRADING-SCORE-N90"]["proof_claim_status"] == "P2"


def test_spot_check_lean_kernel_compilation():
    """Directly compile representative spot-check modules via local Lean binary."""
    _require_provenance_inputs()
    bridge = LeanUniversalBridge(backend=VerificationBackend.CORE_AXIOMATIC)
    if not bridge.is_lean_compiler_available:
        pytest.skip("spot-check compilation requires the Lean 4 compiler")

    specs = assemble_exhaustive_candidate_specifications()
    spot_ids = [
        "MATH-CELL-U15-H15M-HYP5_FOURIER_SPECTRAL",
        "STRUCT-MASTER-SIZING-PHI",
        "GA-D3-CHAMPION-U90_H60M",
        "TRADING-SCORE-N90",
    ]
    target_specs = [s for s in specs if s.candidate_id in spot_ids]
    assert len(target_specs) == 4

    for spec in target_specs:
        res = bridge.verify_candidate_formally(spec, standalone_core=True)
        assert res.kernel_exit_code == 0
        assert res.proof_claim_status == ProofClaimStatus.P2_VERIFIED
        assert res.all_theorems_proven is True
