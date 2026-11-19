"""Tests for GF(2^12) Field Tensor Search and Quantum Discovery Engine.

Verifies mathematical invariants, 4-tier baseline hierarchy, tensor verification,
quantum resource metrics, reproducibility manifests, and invsqrt screening.
"""
import numpy as np

from apedp_dyn.quantum.field_tensor_search import (
    DEGREE,
    InvsqrtNumericalModel,
    build_54_reference,
    build_field_spec,
    build_karatsuba_reference,
    build_multiplication_tensor,
    build_schoolbook_reference,
    compile_reversible_multiplier,
    evaluate_multiplier,
    generate_reproducibility_manifest,
    load_published_51_reference,
    verify_tensor_decomposition,
)


def test_field_spec_invariants():
    """Validates field modulus 4179, root order 4095, and primitivity."""
    spec = build_field_spec()
    assert spec.degree == 12
    assert spec.modulus_int == 4179
    assert spec.modulus_poly == (12, 6, 4, 1, 0)
    assert spec.order == 4095
    assert spec.primitive_element == 2
    assert spec.is_irreducible
    assert spec.is_primitive
    # Binary check: 4096 + 64 + 16 + 2 + 1 = 4179
    assert (1 << 12) + (1 << 6) + (1 << 4) + (1 << 1) + 1 == 4179


def test_multiplication_tensor_properties():
    """Validates exact 12x12x12 tensor and commutativity symmetry."""
    T = build_multiplication_tensor()
    assert T.shape == (12, 12, 12)
    assert T.dtype == np.uint8
    # Test commutativity T_ijk == T_jik
    assert np.array_equal(T, np.transpose(T, (1, 0, 2)))
    # Non-triviality check
    assert np.sum(T) > 0


def test_schoolbook_baseline_144():
    """Verifies naive 144-product decomposition against exact tensor."""
    T = build_multiplication_tensor()
    schoolbook = build_schoolbook_reference()
    assert schoolbook.rank == 144
    is_valid, mismatches = verify_tensor_decomposition(T, schoolbook)
    assert is_valid, f"Schoolbook verification failed with {len(mismatches)} errors"


def test_karatsuba_reference_63():
    """Verifies published 63-product Karatsuba decomposition against exact tensor."""
    T = build_multiplication_tensor()
    karatsuba = build_karatsuba_reference()
    assert karatsuba.rank == 63
    is_valid, mismatches = verify_tensor_decomposition(T, karatsuba)
    assert is_valid, f"Karatsuba verification failed with {len(mismatches)} errors"


def test_composed_karatsuba_54():
    """Verifies 54-product Karatsuba base-case composition against exact tensor."""
    T = build_multiplication_tensor()
    c54 = build_54_reference()
    assert c54.rank == 54
    is_valid, mismatches = verify_tensor_decomposition(T, c54)
    assert is_valid, f"54-term Karatsuba verification failed with {len(mismatches)} errors"


def test_incumbent_hierarchy_correction_a():
    """Correction A: Reconciles Schoolbook 144 > Karatsuba 63 > Composed 54 > Published 51."""
    s144 = build_schoolbook_reference()
    k63 = build_karatsuba_reference()
    c54 = build_54_reference()
    p51 = load_published_51_reference()
    
    assert s144.rank == 144
    assert k63.rank == 63
    assert c54.rank == 54
    assert p51.rank == 51
    assert p51.rank < c54.rank < k63.rank < s144.rank
    assert p51.metadata["category"] == "published_hand_tuned_incumbent"


def test_circuit_cost_independence_correction_b():
    """Correction B: Enforces independent recording of tensor rank vs quantum cost."""
    k63 = build_karatsuba_reference()
    circuit_uncontrolled = compile_reversible_multiplier(k63, controlled=False)
    circuit_controlled = compile_reversible_multiplier(k63, controlled=True)
    
    # Controlled adds 12 Toffolis for accumulator insertion
    assert circuit_uncontrolled.tensor_rank == 63
    assert circuit_uncontrolled.toffoli_count == 63
    assert circuit_controlled.tensor_rank == 63
    assert circuit_controlled.toffoli_count == 75
    
    metrics = evaluate_multiplier(circuit_uncontrolled)
    assert "mathematical_tensor_rank" in metrics
    assert "toffoli_count" in metrics
    assert "cnot_count" in metrics
    assert "ancilla_count" in metrics


def test_reproducibility_manifest():
    """Validates SHA-256 certificate emission."""
    k63 = build_karatsuba_reference()
    circuit = compile_reversible_multiplier(k63)
    manifest = generate_reproducibility_manifest(k63, circuit, seed=123)
    
    assert "manifest_sha256" in manifest
    assert "tensor_sha256" in manifest
    assert manifest["modulus_int"] == 4179
    assert manifest["tensor_rank"] == 63


def test_invsqrt_numerical_model_correction_d():
    """Correction D: Bit-exact integer and fixed-point screening model matching verifier."""
    model = InvsqrtNumericalModel(max_x=1000, precision_bits=24)
    
    # Test point evaluation with exact target phase
    for x in (1, 16, 256, 1000):
        target, _tol = model.reference_values_point(x)
        passed, ratio, margin = model.evaluate_candidate_point(x, target)
        assert passed
        assert ratio <= 1.0
        assert margin >= 0
        
    # Test rejection of perturbed phase exceeding tolerance
    target_1, tol_1 = model.reference_values_point(1)
    bad_phase = (target_1 + tol_1 + 1000) & model.mask64
    passed, ratio, margin = model.evaluate_candidate_point(1, bad_phase)
    assert not passed
    assert ratio > 1.0
    assert margin < 0


def test_algebraic_failure_diagnostics_consistent_and_inconsistent():
    """Verifies that diagnose_gf2_bilinear_system extracts exact diagnostics, nullity, and defect."""
    from apedp_dyn.quantum.field_tensor_search import (
        build_54_reference,
        diagnose_gf2_bilinear_system,
    )
    T = build_multiplication_tensor()
    k54 = build_54_reference()

    # Consistent case: Karatsuba 54
    diag_valid, W_valid = diagnose_gf2_bilinear_system(k54.U, k54.V, T)
    assert diag_valid.is_consistent
    assert diag_valid.defect_count == 0
    assert diag_valid.unmet_parity_equations == 0
    assert len(diag_valid.failing_coordinates) == 0
    assert diag_valid.rank_deficiency == 0
    assert diag_valid.rank_augmented == diag_valid.rank_A
    assert W_valid is not None
    assert "VALID" in diag_valid.to_reasoning_summary()

    # Inconsistent case: random rank 10
    rng = np.random.default_rng(42)
    U_bad = rng.integers(0, 2, size=(10, DEGREE), dtype=np.uint8)
    V_bad = rng.integers(0, 2, size=(10, DEGREE), dtype=np.uint8)
    diag_bad, _W_bad = diagnose_gf2_bilinear_system(U_bad, V_bad, T)
    assert not diag_bad.is_consistent
    assert diag_bad.defect_count > 0
    assert diag_bad.unmet_parity_equations > 0
    assert len(diag_bad.failing_coordinates) == diag_bad.defect_count
    assert diag_bad.rank_deficiency > 0
    assert diag_bad.target_rank == 10
    assert "ALGEBRAIC DEFECT DETECTED" in diag_bad.to_reasoning_summary()

