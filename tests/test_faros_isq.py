# SPDX-License-Identifier: Apache-2.0
"""Unit and integration tests for FAROS-ISQ quantum compilation framework.

Status: audit-only

Tests the complete FAROS-ISQ representation portfolio:
- SABLE-90 Profiler & baseline breakdown (90Q 157,837.85 / affine-opt 157,749.40).
- Error-Budget Engine worst-case phase error & margin calculation.
- R1: Nonuniform piecewise minimax polynomial.
- R2: Prefix table plus residual polynomial.
- R3: Direct phase decomposition.
- R4: Alternative arithmetic forms (Horner, Estrin tree, Bit-serial).
- R5: Hybrid candidate representations.
- Multi-fidelity Pareto tournament (F0 -> F1 -> Pareto front extraction).
- Proposer integration with QuantumCandidateLedger and ExperimentRunner.
- Search orchestrator (run_faros_isq_search).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from apedp_dyn.quantum_compilation.contracts import load_contract
from apedp_dyn.quantum_compilation.faros_isq import (
    ArithmeticForm,
    ErrorBudgetEngine,
    FarosCandidateConfig,
    FarosIsqProposer,
    FarosParetoTournament,
    FidelityLevel,
    R1_NONUNIFORM_CONFIGS,
    R2_PREFIX_RESIDUAL_CONFIGS,
    R3_DIRECT_PHASE_CONFIGS,
    R4_ALT_ARITHMETIC_CONFIGS,
    R5_HYBRID_CONFIGS,
    ALL_FAROS_PORTFOLIO_CONFIGS,
    REFERENCE_COST_90Q,
    REFERENCE_COST_90Q_AFFINE_OPT,
    RepresentationType,
    Sable90Profiler,
    SegmentationStrategy,
    calibrate_faros_circuit,
    compute_segment_boundaries,
    evaluate_r1_nonuniform_minimax,
    evaluate_r2_prefix_residual,
    evaluate_r3_direct_phase,
    evaluate_r4_alt_arithmetic,
    evaluate_r5_hybrid,
    exhaustive_integer_certificate,
    generate_evaluation_subset,
    load_chebyshev_minimax_tables,
    materialize_faros_circuit,
    run_faros_isq_search,
)


def test_sable_90_profiler_and_official_cost():
    """Verify SABLE-90 baseline profile, metrics, and threshold gating."""
    profile = Sable90Profiler.profile_90q_reference()
    assert profile["num_qubits"] == 90
    assert profile["static_toffoli"] == 1449
    assert profile["rotation_count"] == 44
    assert profile["margin_integer_units"] == 73326605
    assert profile["cost_nominal_90q"] == pytest.approx(157837.85, rel=1e-4)
    assert profile["cost_affine_opt_90q"] == pytest.approx(157749.40, rel=1e-4)

    # Subsystems breakdown verified
    subsystems = profile["subsystem_toffolis"]
    assert subsystems["normalization"] == 205
    assert subsystems["qrom_lookup"] == 418
    assert subsystems["horner_arithmetic"] == 636
    assert subsystems["denormalization_and_phasing"] == 96
    assert subsystems["uncomputation"] == 94

    # Gating checks
    assert Sable90Profiler.is_strictly_improving(157740.0, REFERENCE_COST_90Q_AFFINE_OPT)
    assert not Sable90Profiler.is_strictly_improving(157750.0, REFERENCE_COST_90Q_AFFINE_OPT)
    assert Sable90Profiler.is_strictly_improving(157800.0, REFERENCE_COST_90Q)


def test_error_budget_engine_margin_and_tolerance():
    """Verify worst-case phase error and tolerance margin calculation."""
    xs = generate_evaluation_subset(max_x=2048)
    assert xs[0] == 1
    assert xs[-1] == 2048

    # Target and tolerance
    target = ErrorBudgetEngine.target_phase_units(xs)
    tol = ErrorBudgetEngine.tolerance_phase_units(xs)
    assert target.shape == xs.shape
    assert tol.shape == xs.shape
    assert (tol > 0).all()

    # Exact model matches target -> 0 error -> max margin
    min_margin, worst_ratio, worst_x = ErrorBudgetEngine.evaluate_margin(target, xs)
    assert min_margin > 0
    assert worst_ratio == pytest.approx(0.0)

    # Intentionally corrupt phase to exceed tolerance
    corrupted = (target + tol * 2).astype(np.uint64)
    min_margin_c, worst_ratio_c, _ = ErrorBudgetEngine.evaluate_margin(corrupted, xs)
    assert min_margin_c < 0
    assert worst_ratio_c > 1.0


def test_r1_nonuniform_segmentation_and_evaluation():
    """Verify R1 nonuniform segmentation strategies and minimax phase model."""
    xs = generate_evaluation_subset(max_x=2048)

    # Compare uniform vs curvature-equalized boundaries
    b_uni = compute_segment_boundaries(32, SegmentationStrategy.UNIFORM, 1.0, 2.0)
    b_curv = compute_segment_boundaries(32, SegmentationStrategy.CURVATURE_EQUALIZED, 1.0, 2.0)
    assert len(b_uni) == 33
    assert len(b_curv) == 33
    assert b_curv[0] == 1.0
    assert pytest.approx(b_curv[-1], rel=1e-5) == 2.0

    # Curvature equalized should have smaller step size near M=1 than near M=2
    step_low = b_curv[1] - b_curv[0]
    step_high = b_curv[-1] - b_curv[-2]
    assert step_low < step_high

    # Evaluate R1 candidate
    cfg = R1_NONUNIFORM_CONFIGS[0]
    passed, reason, metrics = evaluate_r1_nonuniform_minimax(cfg, xs)
    assert passed is True, reason
    assert metrics["passed_f0"] is True
    assert metrics["worst_error_over_tolerance"] < 1.0
    assert metrics["min_margin_units"] > 0

    cost_eval = metrics["cost_evaluation"]
    assert cost_eval["strictly_below_90q"] is True
    assert cost_eval["strictly_below_affine_opt"] is True
    assert cost_eval["cost"] < REFERENCE_COST_90Q_AFFINE_OPT


def test_r2_prefix_table_residual_polynomial():
    """Verify R2 prefix table plus residual polynomial model and cost."""
    xs = generate_evaluation_subset(max_x=2048)
    cfg = R2_PREFIX_RESIDUAL_CONFIGS[0]  # 6-bit prefix + 11-bit delta

    passed, reason, metrics = evaluate_r2_prefix_residual(cfg, xs)
    assert passed is True, reason
    assert metrics["passed_f0"] is True
    assert metrics["min_margin_units"] > 0

    cost_eval = metrics["cost_evaluation"]
    assert cost_eval["strictly_below_90q"] is True
    assert cost_eval["strictly_below_affine_opt"] is True
    assert cost_eval["num_qubits"] <= 84
    assert cost_eval["cost"] < REFERENCE_COST_90Q_AFFINE_OPT


def test_r3_direct_phase_decomposition():
    """Verify R3 direct phase decomposition evaluation."""
    xs = generate_evaluation_subset(max_x=1024)
    cfg = R3_DIRECT_PHASE_CONFIGS[0]

    passed, reason, metrics = evaluate_r3_direct_phase(cfg, xs)
    assert passed is True, reason
    assert metrics["passed_f0"] is True
    cost_eval = metrics["cost_evaluation"]
    assert cost_eval["num_qubits"] == 74
    assert cost_eval["cost"] < REFERENCE_COST_90Q


def test_r4_alternative_arithmetic_forms():
    """Verify R4 Estrin tree and bit-serial arithmetic evaluation."""
    xs = generate_evaluation_subset(max_x=1024)

    # Estrin tree
    estrin_cfg = [c for c in R4_ALT_ARITHMETIC_CONFIGS if c.arithmetic_form == ArithmeticForm.ESTRIN_TREE][0]
    p_estrin, _, m_estrin = evaluate_r4_alt_arithmetic(estrin_cfg, xs)
    assert p_estrin is True
    assert m_estrin["cost_evaluation"]["num_qubits"] == 86
    assert m_estrin["cost_evaluation"]["strictly_below_affine_opt"] is True

    # Bit-serial
    serial_cfg = [c for c in R4_ALT_ARITHMETIC_CONFIGS if c.arithmetic_form == ArithmeticForm.BIT_SERIAL][0]
    p_serial, _, m_serial = evaluate_r4_alt_arithmetic(serial_cfg, xs)
    assert p_serial is True
    assert m_serial["cost_evaluation"]["num_qubits"] == 78
    assert m_serial["cost_evaluation"]["strictly_below_90q"] is True


def test_r5_hybrid_candidate():
    """Verify R5 hybrid candidate (prefix table + Estrin tree)."""
    xs = generate_evaluation_subset(max_x=1024)
    cfg = R5_HYBRID_CONFIGS[0]

    passed, reason, metrics = evaluate_r5_hybrid(cfg, xs)
    assert passed is True, reason
    cost_eval = metrics["cost_evaluation"]
    assert cost_eval["strictly_below_affine_opt"] is True
    assert cost_eval["cost"] < 152000.0


def test_pareto_dominance_logic():
    """Verify multi-objective Pareto dominance detection."""
    # c1 dominates c2 if all objectives <= and at least one <
    c1 = {"num_qubits": 84, "static_toffoli": 1300, "rotation_count": 40, "cost": 150000.0}
    c2 = {"num_qubits": 86, "static_toffoli": 1350, "rotation_count": 42, "cost": 155000.0}
    assert FarosParetoTournament.is_pareto_dominant(c1, c2) is True
    assert FarosParetoTournament.is_pareto_dominant(c2, c1) is False

    # Trade-off: c3 has fewer qubits but higher cost -> neither dominates
    c3 = {"num_qubits": 78, "static_toffoli": 1500, "rotation_count": 44, "cost": 154000.0}
    assert FarosParetoTournament.is_pareto_dominant(c1, c3) is False
    assert FarosParetoTournament.is_pareto_dominant(c3, c1) is False


def test_multi_fidelity_pareto_tournament_execution():
    """Verify full tournament execution across the portfolio."""
    xs = generate_evaluation_subset(max_x=2048)
    report = FarosParetoTournament.run_tournament(
        candidates=ALL_FAROS_PORTFOLIO_CONFIGS,
        xs=xs,
        max_fidelity=FidelityLevel.F1_REVERSIBLE_COST,
    )

    assert report.total_screened == len(ALL_FAROS_PORTFOLIO_CONFIGS)
    assert report.passed_f0_count > 0
    assert report.strictly_below_90q_count > 0
    assert report.strictly_below_affine_opt_count > 0
    assert len(report.pareto_frontier) > 0

    best = report.best_candidate
    assert best is not None
    assert best.score is not None
    assert best.score < REFERENCE_COST_90Q_AFFINE_OPT
    assert best.is_pareto_optimal is True


def test_faros_isq_proposer_screen_compile_and_measure():
    """Verify the FAROS proposer can screen, compile, and price candidates."""
    contract = load_contract("invsqrt")

    proposer = FarosIsqProposer(portfolio=R2_PREFIX_RESIDUAL_CONFIGS[:2])
    candidates = proposer.generate(contract, incumbent=None, budget=2)
    assert len(candidates) == 2

    # Screen first candidate
    screen = proposer.screen(candidates[0], contract)
    assert screen.valid is True
    assert screen.metrics["min_margin_units"] > 0

    # Compile and measure
    artifact = proposer.compile(candidates[0], contract)
    cost = proposer.measure_resources(artifact, contract)
    assert cost < REFERENCE_COST_90Q_AFFINE_OPT


def test_run_faros_isq_search_end_to_end(tmp_path):
    """Verify end-to-end FAROS search orchestrator and output report."""
    report = run_faros_isq_search(
        output_dir=tmp_path / "faros_run",
        budget=5,
        dev_max_x=1024,
    )

    assert "sable_90_baseline" in report
    assert "tournament_summary" in report
    assert "pareto_frontier" in report
    assert "ledger_summary" in report

    summary = report["tournament_summary"]
    assert summary["total_candidates"] <= 5
    assert summary["passed_f0"] > 0
    assert summary["best_score"] is not None
    assert summary["best_score"] < REFERENCE_COST_90Q_AFFINE_OPT

    # Verify JSON file written to disk
    json_path = tmp_path / "faros_run" / "faros_isq_report.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["tournament_summary"]["best_score"] == summary["best_score"]


def test_faros_f0_rejects_insufficient_precision():
    """Verify that F0 screening rejects configs with insufficient precision."""
    xs = generate_evaluation_subset(max_x=1024)
    # 4 segments with 8-bit coefficients cannot satisfy 2^-24 tolerance
    bad_cfg = FarosCandidateConfig(
        candidate_id="faros-bad-precision",
        representation=RepresentationType.R1_NONUNIFORM_MINIMAX,
        segmentation=SegmentationStrategy.UNIFORM,
        degree=1,
        num_segments=4,
        arithmetic_form=ArithmeticForm.HORNER,
        prefix_bits=0,
        delta_bits=6,
        coef_bits=8,
        output_bits=8,
        num_qubits=70,
    )
    passed, reason, metrics = FarosParetoTournament.evaluate_representation(bad_cfg, xs)
    assert passed is False
    assert reason is not None
    assert ">= 1.0" in reason
    assert metrics["passed_f0"] is False
    assert metrics["worst_error_over_tolerance"] > 1.0


def test_all_faros_portfolio_configs_pass_f0():
    """Verify all 9 standard portfolio candidates pass F0 screen on dev subset."""
    xs = generate_evaluation_subset(max_x=2048)
    for cfg in ALL_FAROS_PORTFOLIO_CONFIGS:
        passed, reason, metrics = FarosParetoTournament.evaluate_representation(cfg, xs)
        assert passed is True, f"{cfg.candidate_id} failed: {reason}"
        assert metrics["min_margin_units"] > 0
        cost_eval = metrics["cost_evaluation"]
        assert cost_eval["strictly_below_affine_opt"] is True
        assert cost_eval["cost"] < REFERENCE_COST_90Q_AFFINE_OPT


def test_sable_90_profiler_negative_inputs():
    """Verify Sable90Profiler.evaluate_cost rejects invalid or negative inputs."""
    with pytest.raises(ValueError, match="strictly positive"):
        Sable90Profiler.evaluate_cost(qubits=0, static_toffoli=1000, rotations=40, margin_integer_units=1000)

    with pytest.raises(ValueError, match="strictly positive"):
        Sable90Profiler.evaluate_cost(qubits=90, static_toffoli=-10, rotations=40, margin_integer_units=1000)

    with pytest.raises(ValueError, match="strictly positive"):
        Sable90Profiler.evaluate_cost(qubits=90, static_toffoli=1000, rotations=0, margin_integer_units=1000)

    with pytest.raises(ValueError, match="strictly positive"):
        Sable90Profiler.evaluate_cost(qubits=90, static_toffoli=1000, rotations=40, margin_integer_units=0)


def test_chebyshev_minimax_table_generation_and_loading():
    """Verify Chebyshev minimax table structure, segment count, and coefficient bounds."""
    cfg, tables, worst = load_chebyshev_minimax_tables()
    assert len(tables) == 256  # 128 segments * 2 parity branches
    assert (cfg.f2, cfg.f1, cfg.fyr, cfg.gy) == (8, 19, 27, 1)
    assert worst * (2**24) < 1.0  # Normalized continuous minimax ratio < 1.0

    # Every triple (K2, K1, K0) must be positive and within bit bounds
    for key, (k2, k1, k0) in tables.items():
        assert 0 <= k2 < (1 << cfg.w2)
        assert k1 > 0
        assert k0 > 0


def test_exhaustive_integer_certificate_execution(tmp_path):
    """Verify exhaustive integer certificate evaluates and verifies within tolerance."""
    cert_file = tmp_path / "test_certificate.json"
    cfg, tables, _ = load_chebyshev_minimax_tables()

    # Run on a chunk size to verify execution and margin properties
    cert = exhaustive_integer_certificate(
        cfg, tables, chunk_size=262144, output_json=cert_file
    )
    assert cert["pass"] is True
    assert cert["tested_inputs"] == 3133452
    assert cert["worst_ratio"] < 1.0
    assert cert["min_margin_units"] > 0
    assert cert_file.exists()

    loaded = json.loads(cert_file.read_text(encoding="utf-8"))
    assert loaded["pass"] is True
    assert loaded["tested_inputs"] == 3133452


def test_materialize_faros_circuit_execution(tmp_path):
    """Verify materialization produces circuit artifact, SHA-256, and metadata."""
    out_kmb = tmp_path / "invsqrt_faros_test_86q.kmb"
    meta = materialize_faros_circuit(
        output_path=out_kmb,
        candidate_id="invsqrt_faros_test_86q",
    )
    assert out_kmb.exists()
    assert meta["path"] == out_kmb
    assert len(meta["sha256"]) == 64
    assert meta["num_qubits"] <= 90
    assert meta["static_toffoli"] == 1448
    assert meta["rotations"] == 44
    assert meta["cost_evaluation"]["cost"] < 160000.0

    # Companion resources file verified
    res_file = out_kmb.with_suffix(".kmb.resources.json")
    assert res_file.exists()


def test_calibrate_faros_circuit_execution(tmp_path):
    """Verify fixed-delta replay without claiming an LP solver ran."""
    out_kmb = tmp_path / "invsqrt_faros_test_cal.kmb"
    materialize_faros_circuit(output_path=out_kmb)

    cal_rep = calibrate_faros_circuit(out_kmb)
    assert cal_rep["calibrated_rotation_count"] == 44
    assert cal_rep["margin_units"] == 73326605
    assert cal_rep["lp_converged"] is None
    assert cal_rep["margin_verified"] is False
    assert "dropped_original_rotation_indices" in cal_rep


def test_verify_faros_candidate_receipt_structure(tmp_path):
    """Verify receipt structure and contract verification for FAROS candidate."""
    receipt_file = Path("artifacts/quantum_compilation/invsqrt/receipt_faros_r2_86q.json")
    assert receipt_file.exists()
    data = json.loads(receipt_file.read_text(encoding="utf-8"))

    assert data["status"] == "PASS"
    assert data["benchmark"] == "invsqrt"
    assert data["verified_shots"] >= 200540928
    assert data["worst_error_over_tolerance"] < 1.0
    assert data["circuit_sha256"] == "7bd8955e7bda35382f33ba6df6239c746bcb1adad627f585939db6bc1accc3d0"
