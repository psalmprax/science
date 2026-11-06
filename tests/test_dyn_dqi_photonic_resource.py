"""Tests for the DQI -> sFFCC deterministic-photonics resource mapping.

Status: audit-only

What this suite pins down:
1. Per-patch sFFCC resource counts follow the blueprint formulas exactly
   (6L^2 emitters, 18L^2 detectors, 24L^2 active elements with EBF, 18L^3 fusions).
2. Totals scale with the DQI logical patch count Q.
3. Fusion retry accounting: success = eta^2, attempts = fusions / success.
4. The Benchmark II score is platform-invariant: identical at every code
   distance, while wall-clock scales linearly in L and emitters quadratically.
   This is the finding that blocks "photonic advantage" claims on the score.
5. The Gluzdov V6 row is carried as an unverified target, never as a receipt.
6. The mapper never claims a compilation or a verifier run.
"""

from __future__ import annotations

import math

import pytest

from apedp_dyn.quantum.dqi_photonic_resource import (
    DEFAULT_CODE_DISTANCE,
    DQI_GLUZDOV_V6_TARGET_WORKLOAD,
    DQI_REFERENCE_WORKLOAD,
    DQIPhotonicMappingAssumptions,
    DQIPhotonicResourceMapper,
    DQIWorkload,
    SuperconductingBaseline,
    map_reference_workload,
)
from apedp_dyn.quantum.sparrow_photonics import SparrowHardwareSpec

R_EPS_PLUS = 1.1912604648115832


def test_per_patch_resources_follow_blueprint_formulas():
    mapping = DQIPhotonicResourceMapper().map_workload(DQI_REFERENCE_WORKLOAD)
    per_patch = mapping["per_patch_resources"]
    L = DEFAULT_CODE_DISTANCE

    assert per_patch["code_distance_L"] == L
    assert per_patch["emitters"] == 6 * L * L
    assert per_patch["photodetectors"] == 18 * L * L
    assert per_patch["active_elements_with_ebf"] == 24 * L * L
    assert per_patch["encoded_fusions_per_cycle"] == 18 * L ** 3
    # EBF removes 9L^2 - 4L + 1 active phase shifters.
    assert per_patch["active_elements_saved_by_ebf"] == 9 * L * L - 4 * L + 1


def test_hardware_totals_scale_with_logical_patch_count():
    mapper = DQIPhotonicResourceMapper()
    mapping = mapper.map_workload(DQI_REFERENCE_WORKLOAD)
    totals = mapping["hardware_totals"]
    per_patch = mapping["per_patch_resources"]
    Q = DQI_REFERENCE_WORKLOAD.qubits

    assert totals["logical_patches"] == Q
    assert totals["emitters"] == per_patch["emitters"] * Q
    assert totals["photodetectors"] == per_patch["photodetectors"] * Q
    assert totals["active_elements"] == per_patch["active_elements_with_ebf"] * Q
    assert totals["fusions_per_cycle"] == per_patch["encoded_fusions_per_cycle"] * Q


def test_fusion_retry_accounting():
    mapper = DQIPhotonicResourceMapper()
    mapping = mapper.map_workload(DQI_REFERENCE_WORKLOAD)
    totals = mapping["hardware_totals"]

    eta = SparrowHardwareSpec().eta_end_to_end
    assert math.isclose(totals["fusion_success_probability"], eta * eta, rel_tol=1e-12)
    assert math.isclose(
        totals["expected_fusion_attempts_per_shot"],
        totals["fusions_per_shot"] / (eta * eta),
        rel_tol=1e-12,
    )
    # A retry factor can only add work, never remove it.
    assert totals["expected_fusion_attempts_per_shot"] > totals["fusions_per_shot"]


def test_score_is_platform_invariant_while_wall_clock_scales_with_L():
    scores = []
    wall_clocks = []
    emitters = []
    for L in (8, DEFAULT_CODE_DISTANCE, 24):
        mapper = DQIPhotonicResourceMapper(
            assumptions=DQIPhotonicMappingAssumptions(code_distance=L)
        )
        mapping = mapper.map_workload(DQI_REFERENCE_WORKLOAD, r_overhead=R_EPS_PLUS)
        scores.append(mapping["score_invariance"]["official_score"])
        wall_clocks.append(mapping["timing"]["wall_clock_seconds_per_shot"])
        emitters.append(mapping["hardware_totals"]["emitters"])

    # The official score has no timing term, so the code distance cannot move it.
    assert math.isclose(scores[0], scores[1], rel_tol=1e-12)
    assert math.isclose(scores[1], scores[2], rel_tol=1e-12)
    assert mapping["score_invariance"]["contains_timing_term"] is False
    assert mapping["score_invariance"]["platform_invariant_given_same_Q_and_T"] is True

    # ... but the bill does move: linear in L for time, quadratic for emitters.
    assert wall_clocks[2] / wall_clocks[0] == pytest.approx(3.0, rel=1e-9)
    assert emitters[2] / emitters[0] == pytest.approx((24 / 8) ** 2, rel=1e-9)


def test_official_score_reproduces_sealed_baseline_within_rounding():
    mapping = DQIPhotonicResourceMapper().map_workload(
        DQI_REFERENCE_WORKLOAD, r_overhead=R_EPS_PLUS
    )
    score = mapping["score_invariance"]["official_score"]
    # The workload carries T_mean rounded to two decimals from the receipt, so the
    # recomputed score matches the sealed 24,439,463,583,672.93 only to ~1e-9.
    assert math.isclose(score, 24_439_463_583_672.93, rel_tol=1e-9)


def test_gluzdov_v6_row_is_a_target_not_a_receipt():
    assert DQI_REFERENCE_WORKLOAD.has_compiled_receipt is True
    assert DQI_GLUZDOV_V6_TARGET_WORKLOAD.has_compiled_receipt is False

    mapper = DQIPhotonicResourceMapper()
    comparison = mapper.compare_workloads(
        DQI_REFERENCE_WORKLOAD,
        DQI_GLUZDOV_V6_TARGET_WORKLOAD,
        r_overhead=R_EPS_PLUS,
    )

    assert comparison["candidate_has_compiled_receipt"] is False
    assert comparison["reasons_to_distrust"]
    assert comparison["projected_gain_factor"] > 14.0
    assert comparison["logical_cost_ratio"] < 1.0
    # Photonic emitter count tracks the qubit ratio exactly (same code distance).
    assert comparison["emitter_ratio"] == pytest.approx(
        DQI_GLUZDOV_V6_TARGET_WORKLOAD.qubits / DQI_REFERENCE_WORKLOAD.qubits
    )


def test_mapping_never_claims_compilation_or_verification():
    mapping = DQIPhotonicResourceMapper().map_workload(
        DQI_GLUZDOV_V6_TARGET_WORKLOAD, r_overhead=R_EPS_PLUS
    )
    assert mapping["status"] == "ANALYTICAL_PROJECTION_ONLY"
    assert mapping["compiled_kmb"] is False
    assert mapping["official_verifier_run"] is False
    assert mapping["promotion_decision"] == "HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT"


def test_eea_arithmetic_decomposition():
    mapping = DQIPhotonicResourceMapper().map_workload(DQI_REFERENCE_WORKLOAD)
    eea = mapping["eea_arithmetic_mapping"]
    workload = DQI_REFERENCE_WORKLOAD

    assert eea["gf2_word_bits"] == 12
    assert eea["polynomial_registers"] == workload.syndromes + 3
    assert eea["iterations"] == workload.eea_iterations
    assert math.isclose(
        eea["toffoli_like_per_iteration"],
        workload.mean_toffoli_like / workload.eea_iterations,
        rel_tol=1e-12,
    )
    assert math.isclose(
        mapping["timing"]["cycles_per_eea_iteration"],
        mapping["timing"]["cycles_per_shot"] / workload.eea_iterations,
        rel_tol=1e-12,
    )


def test_fault_tolerance_audit_reports_channel_pass_and_budget_failure():
    """Per-channel thresholds pass, but the consolidated noise budget does not."""
    mapping = DQIPhotonicResourceMapper().map_workload(DQI_REFERENCE_WORKLOAD)
    ft = mapping["fault_tolerance"]

    assert ft["all_channels_passed"] is True
    # Sum of per-channel loads is ~4.97, above the unit budget, so the referee
    # withholds fault tolerance even though no single channel is violated.
    assert ft["normalized_noise_budget_load"] > 1.0
    assert ft["is_fault_tolerant"] is False
    assert ft["meets_reference_logical_error_rate"] is True


def test_fault_tolerance_rejects_excess_loss():
    mapping = DQIPhotonicResourceMapper(
        hardware=SparrowHardwareSpec(eta_end_to_end=0.88)
    ).map_workload(DQI_REFERENCE_WORKLOAD)
    assert mapping["fault_tolerance"]["all_channels_passed"] is False


def test_reference_workload_convenience_wrapper():
    mapping = map_reference_workload(code_distance=10)
    assert mapping["per_patch_resources"]["code_distance_L"] == 10
    assert mapping["workload"]["label"] == DQI_REFERENCE_WORKLOAD.label


def test_platform_comparison_wall_clock_and_composition():
    """Wall-clock and hardware numbers follow from the stated baseline assumptions."""
    mapper = DQIPhotonicResourceMapper()
    baseline = SuperconductingBaseline(
        logical_cycle_seconds=1e-6,
        physical_qubits_per_logical=1000.0,
    )
    cmp = mapper.compare_platforms(DQI_REFERENCE_WORKLOAD, baseline, r_overhead=R_EPS_PLUS)

    cycles = cmp["wall_clock"]["logical_cycles"]
    assert cycles == cmp["wall_clock"]["logical_cycles"]
    assert cmp["wall_clock"]["superconducting_seconds_per_shot"] == pytest.approx(
        cycles * 1e-6, rel=1e-12
    )
    assert cmp["wall_clock"]["photonic_over_superconducting"] == pytest.approx(
        cmp["wall_clock"]["photonic_seconds_per_shot"]
        / cmp["wall_clock"]["superconducting_seconds_per_shot"],
        rel=1e-12,
    )
    assert cmp["hardware_composition"]["superconducting_physical_qubits"] == (
        DQI_REFERENCE_WORKLOAD.qubits * 1000.0
    )
    # The baseline is an assumption, so it must be disclosed as one.
    assert "caller-supplied" in cmp["assumptions_disclosure"]["superconducting"]
    assert cmp["score_comparison"]["applicable"] is False
    assert cmp["official_verifier_run"] is False


def test_superconducting_baseline_requires_stated_assumptions():
    with pytest.raises(TypeError):
        SuperconductingBaseline()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="logical_cycle_seconds must be positive"):
        SuperconductingBaseline(logical_cycle_seconds=0.0, physical_qubits_per_logical=1000.0)
    with pytest.raises(ValueError, match="physical_qubits_per_logical must be positive"):
        SuperconductingBaseline(logical_cycle_seconds=1e-6, physical_qubits_per_logical=0.0)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError, match="qubits must be positive"):
        DQIWorkload(label="bad", qubits=0, mean_toffoli_like=1.0)
    with pytest.raises(ValueError, match="mean_toffoli_like must be positive"):
        DQIWorkload(label="bad", qubits=10, mean_toffoli_like=0.0)
    with pytest.raises(ValueError, match="eea_iterations must be positive"):
        DQIWorkload(label="bad", qubits=10, mean_toffoli_like=1.0, eea_iterations=0)
    with pytest.raises(ValueError, match="code_distance must be at least 3"):
        DQIPhotonicMappingAssumptions(code_distance=2)
    with pytest.raises(ValueError, match="logical_ops_per_cycle must be positive"):
        DQIPhotonicMappingAssumptions(logical_ops_per_cycle=0.0)
    with pytest.raises(ValueError, match="fusion_success_probability"):
        DQIPhotonicMappingAssumptions(fusion_success_probability=0.0)
    with pytest.raises(ValueError, match="r_overhead must be positive"):
        DQIPhotonicResourceMapper().map_workload(DQI_REFERENCE_WORKLOAD, r_overhead=0.0)
