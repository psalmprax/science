"""Tests for MMLM-POC-001 Matched-Budget Benchmark Runner.

Verifies the 3-arm benchmark execution, budget compliance, 144-constraint solver,
ledger emission, pre-registered outcome classification, 4-budget accountant,
and epistemic gate auditing.
"""
import json
import pathlib
import tempfile
import numpy as np
import pytest

from apedp_dyn.quantum.field_tensor_search import (
    DEGREE,
    build_54_reference,
    build_multiplication_tensor,
    solve_gf2_bilinear_weights,
    verify_tensor_decomposition,
)
from scripts.run_mmlm_poc_001 import (
    BudgetAccountant,
    audit_epistemic_launch_gates,
    run_arm_a_conventional,
    run_arm_b_mmlm,
    run_arm_c_algebraic,
    run_mmlm_poc_001_matched_benchmark,
)


def test_144_constraint_solver_consistency():
    """Verifies that the GF(2) linear solver correctly solves for W given valid (U, V)."""
    T = build_multiplication_tensor()
    k54 = build_54_reference()
    
    # Solve W given U and V from the verified 54-term decomposition
    success, W, defects = solve_gf2_bilinear_weights(k54.U, k54.V, T)
    assert success
    assert defects == 0
    assert W is not None
    assert W.shape == (54, DEGREE)
    
    # Verify the solved W produces exact tensor reconstruction
    k54_reconstructed = type(k54)(
        rank=54, U=k54.U, V=k54.V, W=W, name="K54-Reconstructed"
    )
    is_valid, mismatches = verify_tensor_decomposition(T, k54_reconstructed)
    assert is_valid
    assert len(mismatches) == 0


def test_144_constraint_solver_detects_inconsistent_subspace():
    """Verifies that the GF(2) linear solver detects when (U, V) cannot span T."""
    T = build_multiplication_tensor()
    # Random rank 10 matrices cannot span the 144 constraints of GF(2^12) multiplication
    rng = np.random.default_rng(1234)
    U_bad = rng.integers(0, 2, size=(10, DEGREE), dtype=np.uint8)
    V_bad = rng.integers(0, 2, size=(10, DEGREE), dtype=np.uint8)
    
    success, W, defects = solve_gf2_bilinear_weights(U_bad, V_bad, T)
    assert not success
    assert defects > 0
    assert W is None


def test_budget_accountant_limits_and_exhaustion():
    """Verifies that BudgetAccountant rigorously tracks limits and halts execution."""
    accountant = BudgetAccountant(
        wall_clock_limit_sec=10.0,
        cpu_time_limit_sec=10.0,
        max_evaluations=5,
        max_verifier_calls=5,
        max_tokens=1000,
        arm_name="TestArm",
    )
    assert not accountant.is_exhausted()
    accountant.record_eval(3)
    accountant.record_verifier_call(3)
    accountant.record_tokens(500)
    assert not accountant.is_exhausted()

    # Exceed evaluations
    accountant.record_eval(3)
    assert accountant.is_exhausted()
    assert accountant.exhaustion_reason == "max_evaluations_exceeded"

    summary = accountant.to_dict()
    assert summary["arm"] == "TestArm"
    assert summary["evaluations_count"] == 6
    assert summary["tokens_count"] == 500


def test_launch_gate_audit_enforces_gate_0_hard_stop():
    """Verifies that audit_epistemic_launch_gates strictly enforces Gate 0 hard stop."""
    T = build_multiplication_tensor()
    gate_audit = audit_epistemic_launch_gates(T)
    
    # Gate 0 must trigger Hard Stop because Khattar et al. (2025) matrices are not on disk
    assert not gate_audit["gate_0_hard_stop"]["cleared"]
    assert gate_audit["gate_0_hard_stop"]["status"] == "TRIGGERED_HARD_STOP"
    
    # Gate 1 must clear because 144, 63, 54 are verified
    assert gate_audit["gate_1_baseline_hierarchy"]["cleared"]
    assert gate_audit["gate_1_baseline_hierarchy"]["schoolbook_144_valid"]
    assert gate_audit["gate_1_baseline_hierarchy"]["published_karatsuba_63_valid"]
    assert gate_audit["gate_1_baseline_hierarchy"]["composed_karatsuba_54_valid"]


def test_matched_budget_benchmark_execution():
    """Verifies execution of MMLM-POC-001 across all 3 arms with ledger emission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = pathlib.Path(tmpdir) / "test_ledger.jsonl"
        summary = run_mmlm_poc_001_matched_benchmark(
            budget_sec_per_arm=0.5,
            max_steps_per_arm=20,
            target_rank=50,
            seed=999,
            ledger_path=str(ledger_file),
        )
        
        assert summary["protocol"] == "MMLM-POC-001"
        assert summary["target_rank"] == 50
        assert summary["incumbent_rank"] == 51
        assert summary["incumbent_on_disk_rank"] == 54
        assert summary["outcome"] in ("OUTCOME_1", "OUTCOME_2", "OUTCOME_3", "OUTCOME_4")
        
        # Verify 3 adjudication dimensions
        adj = summary["adjudication_dimensions"]
        assert "dimension_1_mathematical_tensor_rank_sub_51" in adj
        assert "dimension_2_reversible_circuit_toffoli_sub_51" in adj
        assert "dimension_3_arm_b_advantage" in adj
        dim3 = adj["dimension_3_arm_b_advantage"]
        assert "status" in dim3
        assert "replicated" in dim3
        assert "asymmetric_resource_accounting" in dim3
        assert "arm_b_tokens" in dim3["asymmetric_resource_accounting"]

        # Verify launch gate audit
        assert "launch_gate_audit" in summary
        assert not summary["launch_gate_audit"]["gate_0_hard_stop"]["cleared"]

        # Verify ledger file written
        assert ledger_file.exists()
        lines = ledger_file.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["protocol"] == "MMLM-POC-001"
        assert "arm_a_result" in record
        assert "arm_b_result" in record
        assert "arm_c_result" in record
        assert "diagnostics" in record["arm_b_result"]


def test_multi_seed_benchmark_execution():
    """Verifies multi-seed benchmark execution and replication status transition."""
    from scripts.run_mmlm_poc_001 import run_mmlm_poc_001_multi_seed_benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = pathlib.Path(tmpdir) / "test_multi_ledger.jsonl"
        summary = run_mmlm_poc_001_multi_seed_benchmark(
            seeds=(42, 101),
            budget_sec_per_arm=0.5,
            max_steps_per_arm=20,
            target_rank=50,
            ledger_path=str(ledger_file),
        )
        assert summary["protocol"] == "MMLM-POC-001-MULTI-SEED"
        assert summary["seeds_evaluated"] == [42, 101]
        assert "replicated_status" in summary
        assert summary["win_loss_tie_vs_arm_a"]["wins"] >= 0
        assert "asymmetric_resource_totals" in summary
        assert "total_arm_b_reasoning_tokens" in summary["asymmetric_resource_totals"]
