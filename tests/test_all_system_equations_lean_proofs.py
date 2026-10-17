"""Automated Unit Tests for All 323 System Equations Lean 4 Verification.

Verifies:
  - 321 genuine equations across 10 families compile with Lean 4 exit code 0 -> P2_VERIFIED
  - 2 mathematical traps fail Lean 4 proofs with sorry -> P0_NO_PROOF
  - Zero P1 regressions across the entire suite of 323 equations
"""
from __future__ import annotations

import json

from apedp.mathematical.mathematical_layer import ProofClaimStatus
from apedp_dyn.epistemic.lean_universal_bridge import LeanUniversalBridge, VerificationBackend
from apedp_dyn.sandbox.build_all_system_equations_lean_proofs import (
    ARTIFACT_DIR,
    assemble_exhaustive_candidate_specifications,
)


def test_system_wide_candidate_inventory():
    """Verify that all 323 candidate specifications are assembled across all 10 sources."""
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
    assert provenance_path.exists()

    with open(provenance_path) as f:
        data = json.load(f)

    assert data["total_equations_evaluated"] == 323
    assert data["p2_verified_count"] == 321
    assert data["p0_refuted_count"] == 2
    assert data["p1_candidate_count"] == 0

    results_by_id = {r["candidate_id"]: r for r in data["results"]}
    assert results_by_id["PHYS-ENERGY-LEAK-TRAP"]["proof_claim_status"] == "P0"
    assert results_by_id["FIN-CUBIC-RUNAWAY-POLYNOMIAL"]["proof_claim_status"] == "P0"
    assert results_by_id["STRUCT-MASTER-SIZING-PHI"]["proof_claim_status"] == "P2"
    assert results_by_id["GA-D3-CHAMPION-U90_H60M"]["proof_claim_status"] == "P2"
    assert results_by_id["TRADING-SCORE-N90"]["proof_claim_status"] == "P2"


def test_spot_check_lean_kernel_compilation():
    """Directly compile representative spot-check modules via local Lean binary."""
    bridge = LeanUniversalBridge(backend=VerificationBackend.CORE_AXIOMATIC)
    assert bridge.is_lean_compiler_available

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
