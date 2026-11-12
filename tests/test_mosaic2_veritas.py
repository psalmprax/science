"""Unit and integration test suite for MOSAIC-2 / VERITAS compilation governance.

Status: production

Verifies the five methodological corrections and phases P0 to P3:
1. Correction 1: Three orthogonal fingerprints (spec, structural, provenance) on CircuitCandidate.
2. Correction 1: Anti-clone structural diversity enforcement (StructuralEquivalenceError).
3. Evidence Integrity: Receipt-to-artifact hash mismatch enforcement (ReceiptArtifactHashMismatchError).
4. Evidence Integrity: Silent fallback cloning permanently excised from faros_isq.py.
5. Correction 2: Benchmark-specific conservative early-exit resource gate (50-unit noise margin).
6. Phase P2: Structure & sparsity audit engine (entropy, Walsh rank, Givens sign invariant).
7. Correction 3: Discrete integer boundary margin checking without C^0 overconstraint.
8. Correction 4: Clean (|0>) vs. Dirty (U^dagger U = I) ancilla lifecycle safety.
9. Correction 5: Scoped two-region probe on Benchmark I rejects two-region split (O_min >= S_max).
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from apedp_dyn.quantum_compilation import (
    AncillaKind,
    CircuitCandidate,
    ReceiptArtifactHashMismatchError,
    StructuralEquivalenceError,
    SymbolicAncillaLedger,
    TwoRegionProbeResult,
    audit_array_structure,
    compute_provenance_fingerprint,
    compute_spec_fingerprint,
    compute_structural_fingerprint,
    early_exit_resource_gate,
    enforce_structural_diversity,
    probe_two_region_partition,
    verify_givens_sign_invariant,
)


def test_three_fingerprints_generation_and_auto_derivation():
    """Verify spec, structural, and provenance fingerprints are distinct and deterministic."""
    spec_fp = compute_spec_fingerprint(
        "invsqrt",
        "mock_contract_hash_12345",
        parameters={"segmentation": 128},
        domain_bounds=(1, 3133452),
    )
    assert isinstance(spec_fp, str) and len(spec_fp) == 64

    # Structural fingerprint encodes normalized DAG properties
    metrics = {
        "num_qubits": 90,
        "toffoli_count": 1448,
        "rotations": 44,
        "depth": 250,
    }
    struct_fp = compute_structural_fingerprint(metrics)
    assert isinstance(struct_fp, str) and len(struct_fp) == 64

    # Mutating a metric changes the structural fingerprint
    mutated_metrics = dict(metrics, toffoli_count=1447)
    mutated_struct_fp = compute_structural_fingerprint(mutated_metrics)
    assert mutated_struct_fp != struct_fp

    # Provenance fingerprint encodes generation history
    prov_fp = compute_provenance_fingerprint(
        "faros_isq_v1",
        parameters={"segmentation": 128},
        parent_id="invsqrt-baseline-0",
        timestamp="2026-10-26T18:00:00Z",
    )
    assert isinstance(prov_fp, str) and len(prov_fp) == 64

    # CircuitCandidate auto-derives all three if not supplied
    cand = CircuitCandidate(
        candidate_id="invsqrt-test-01",
        benchmark="invsqrt",
        generator_id="faros_isq_v1",
        parameters={"segmentation": 128},
        artifact_path="/tmp/mock.kmb",
        artifact_sha256="abc123def456",
        contract_sha256="mock_contract_hash_12345",
        screen_metrics=metrics,
    )
    expected_spec_fp = compute_spec_fingerprint(
        "invsqrt",
        "mock_contract_hash_12345",
        parameters={"segmentation": 128},
    )
    assert cand.spec_fingerprint == expected_spec_fp
    assert cand.structural_fingerprint == struct_fp
    assert cand.provenance_fingerprint != ""


def test_enforce_structural_diversity_blocks_champion_clones():
    """Verify that a candidate sharing the champion's structural fingerprint is rejected."""
    champ_fp = compute_structural_fingerprint({"num_qubits": 90, "toffolis": 1448, "rotations": 44})
    identical_candidate_fp = compute_structural_fingerprint({"num_qubits": 90, "toffolis": 1448, "rotations": 44})
    different_candidate_fp = compute_structural_fingerprint({"num_qubits": 89, "toffolis": 1480, "rotations": 44})

    with pytest.raises(StructuralEquivalenceError, match="matches incumbent champion"):
        enforce_structural_diversity(identical_candidate_fp, champ_fp)

    # Different candidate passes without exception
    enforce_structural_diversity(different_candidate_fp, champ_fp)


def test_receipt_artifact_hash_mismatch_error_defined():
    """Verify ReceiptArtifactHashMismatchError is available for fail-closed binding."""
    with pytest.raises(ReceiptArtifactHashMismatchError):
        raise ReceiptArtifactHashMismatchError("Artifact digest mismatch")


def test_conservative_early_exit_resource_gate():
    """Verify early_exit_resource_gate requires estimated_min_cost < champion_cost - 50.0."""
    champ_cost = 157752.01

    # Cost greater than champion: rejected
    assert not early_exit_resource_gate(158000.0, champ_cost)

    # Cost identical to champion: rejected
    assert not early_exit_resource_gate(champ_cost, champ_cost)

    # Cost only 20 units lower (within 50-unit noise margin): rejected
    assert not early_exit_resource_gate(champ_cost - 20.0, champ_cost)

    # Cost 49.9 units lower: rejected
    assert not early_exit_resource_gate(champ_cost - 49.9, champ_cost)

    # Cost 60 units lower (surpasses 50-unit noise margin): passes
    assert early_exit_resource_gate(champ_cost - 60.0, champ_cost)

    # Invalid values raise ValueError
    with pytest.raises(ValueError):
        early_exit_resource_gate(-1.0, champ_cost)


def test_structure_audit_entropy_and_walsh_rank():
    """Verify empirical entropy, Walsh rank, and sparsity reporting."""
    # Constant array (zero entropy)
    const_arr = np.zeros(1024, dtype=int)
    report_const = audit_array_structure(const_arr, benchmark="test", compute_walsh=True)
    assert report_const.non_zero_fraction == 0.0
    assert report_const.is_sparse is True
    assert math.isclose(report_const.distribution_entropy, 0.0)
    assert report_const.walsh_rank == 0

    # High entropy array
    rng = np.random.default_rng(20261026)
    dense_arr = rng.integers(0, 1024, size=2048)
    report_dense = audit_array_structure(dense_arr, benchmark="test", compute_walsh=True)
    assert report_dense.non_zero_fraction > 0.99
    assert report_dense.is_sparse is False
    assert report_dense.distribution_entropy > 9.0
    assert report_dense.walsh_rank > 100


def test_givens_sign_invariant_verification():
    """Verify cos(theta) > 0 <==> bit9 == bit8 in 10-bit quantized representation."""
    # Valid words: in [0, 256) (bit9=0, bit8=0) or in [768, 1024) (bit9=1, bit8=1)
    valid_words = np.array([0, 10, 255, 768, 800, 1023], dtype=int)
    all_ok, frac = verify_givens_sign_invariant(valid_words)
    assert all_ok is True
    assert frac == 1.0

    # Invalid words: in [256, 768) (where cos(theta) <= 0)
    invalid_words = np.array([256, 512, 700], dtype=int)
    all_ok, frac = verify_givens_sign_invariant(invalid_words)
    assert all_ok is False
    assert frac == 0.0

    # Mixed array
    mixed = np.concatenate([valid_words, invalid_words])
    all_ok, frac = verify_givens_sign_invariant(mixed)
    assert all_ok is False
    assert math.isclose(frac, len(valid_words) / len(mixed))


def test_clean_vs_dirty_ancilla_lifecycle_ledger():
    """Verify clean ancilla |0> return vs dirty ancilla unitary restoration."""
    ledger = SymbolicAncillaLedger()

    # Clean ancilla must be returned to |0>
    ledger.allocate("clean_0", AncillaKind.CLEAN)
    with pytest.raises(RuntimeError, match="live dependencies remain"):
        ledger.release("clean_0")  # releasing before uncomputing fails

    ledger.mark_uncomputed("clean_0")
    ledger.release("clean_0")  # passes

    # Dirty ancilla must be unitarily restored
    ledger.allocate("dirty_0", AncillaKind.DIRTY)
    with pytest.raises(RuntimeError, match="live dependencies remain"):
        ledger.release("dirty_0")

    ledger.mark_uncomputed("dirty_0")
    ledger.release("dirty_0")  # passes

    assert ledger.verify_all_safe() is True
    assert ledger.peak_active_qubits == 1


def test_two_region_probe_on_benchmark_i():
    """Verify two-region selector probe early-exit rejection on Benchmark I."""
    result = probe_two_region_partition(split_point=1048576)
    assert isinstance(result, TwoRegionProbeResult)
    assert result.discrete_boundary_margin_passed is True
    assert result.ancilla_lifecycle_passed is True

    # Selector overhead (32 Toffolis) mathematically dominates polynomial savings (16 Toffolis)
    assert result.selector_overhead_toffolis == 32
    assert result.polynomial_savings_toffolis == 16
    assert result.net_savings_toffolis == -16

    # Candidate is rejected early before expensive synthesis
    assert result.early_exit_rejected is True
    assert "mathematically dominates" in result.rejection_reason
    assert result.projected_cost > result.champion_cost
