"""Test Suite for Deterministic Quantum Photonics & Fault-Tolerant Quantum Computing (Sparrow Quantum Blueprint).

Status: experimental

Verifies:
1. Probabilistic (SPDC) vs Deterministic (QD) multi-photon resource scaling law.
2. Passive fusion gate and Excitation-Based Feedback (EBF) timing inequality (Eq. 7).
3. sFFCC topological cluster lattice hardware scaling and microsecond logical clock cycle.
4. Epistemic Referee multi-channel audit against Table 2 fault-tolerance thresholds.
"""
from __future__ import annotations

from apedp_dyn.quantum import (
    DeterministicVsProbabilisticScaling,
    PassiveFusionEngine,
    SFFCCLatticeArchitect,
    SparrowFaultToleranceReferee,
    SparrowHardwareSpec,
)


def test_deterministic_vs_probabilistic_scaling():
    # For small n=2 photons, both are comparable
    ratio_2 = DeterministicVsProbabilisticScaling.compute_scaling_ratio(n_photons=2)
    assert ratio_2 > 1.0

    # For n=8 entangled photons, deterministic QD has >10^3x lower physical overhead
    ratio_8 = DeterministicVsProbabilisticScaling.compute_scaling_ratio(n_photons=8)
    assert ratio_8 > 1e3

    # For n=16 photons, SPDC suffers exponential overhead explosion (>10^7x ratio)
    ratio_16 = DeterministicVsProbabilisticScaling.compute_scaling_ratio(n_photons=16)
    assert ratio_16 > 1e7

    # QD overhead scales linearly/polynomially with n
    qd_8 = DeterministicVsProbabilisticScaling.calculate_quantum_dot_overhead(8, eta_out=0.93)
    qd_16 = DeterministicVsProbabilisticScaling.calculate_quantum_dot_overhead(16, eta_out=0.93)
    assert qd_16 < 100.0 * qd_8


def test_passive_fusion_engine_and_ebf_timing():
    spec = SparrowHardwareSpec(
        tau_int=11.83e-9,
        tau_pi=4.0e-9,
        tau_tb=0.5e-9,
        eta_end_to_end=0.93,
    )
    engine = PassiveFusionEngine(spec)

    # Valid EBF delay: 5.0 ns satisfies 0.5 ns < 5.0 ns < (11.83 - 4.0 = 7.83 ns)
    valid, details = engine.verify_ebf_timing_inequality(tau_ebf=5.0e-9)
    assert valid is True
    assert details["headroom_ns"] > 7.0

    # Invalid EBF delay (too short, overlaps with emitter lifetime)
    invalid_short, _ = engine.verify_ebf_timing_inequality(tau_ebf=0.2e-9)
    assert invalid_short is False

    # Invalid EBF delay (too long, arrives after spin pi-pulse)
    invalid_long, _ = engine.verify_ebf_timing_inequality(tau_ebf=9.0e-9)
    assert invalid_long is False

    # Test Type-II fusion at zero offset (entangling Bell-state measurement)
    res_bell = engine.simulate_type2_fusion(input_state="|Psi+>", tau_offset=0.0)
    assert res_bell["mode"] == "two_qubit_bell_fusion"
    assert res_bell["fusion_success_probability"] > 0.40  # Close to theoretical 50% * eta^2

    # Test fusion with EBF offset (deterministic passive single-qubit Z-measurement)
    res_ebf = engine.simulate_type2_fusion(input_state="|Psi+>", tau_offset=5.0e-9)
    assert res_ebf["mode"] == "passive_single_qubit_z"
    assert res_ebf["fusion_success_probability"] > 0.85  # eta^2 without 50% Bell branching penalty!
    assert res_ebf["is_purely_passive"] is True


def test_sffcc_lattice_hardware_scaling():
    spec = SparrowHardwareSpec(tau_rep=13.8e-9)
    
    # Target code distance L=16 (to reach P_L <= 10^-6)
    res_16 = SFFCCLatticeArchitect.calculate_resources(code_distance_l=16, spec=spec)
    
    # 6 * L^2 = 1,536 QD emitters per logical qubit
    assert res_16["num_qd_emitters"] == 1536
    assert res_16["num_photodetectors"] == 18 * (16 ** 2)
    
    # Active elements saved by EBF: (33L^2 - 4L + 1) - 24L^2 = 9L^2 - 4L + 1
    expected_savings = 9 * (16 ** 2) - 4 * 16 + 1
    assert res_16["active_elements_saved_by_ebf"] == expected_savings
    assert res_16["active_elements_saved_by_ebf"] == 2241

    # Optical depth with EBF
    assert res_16["optical_depth_active"] == 4
    
    # Encoded fusions per cycle: 18 * L^3 = 73,728
    assert res_16["encoded_fusions_per_cycle"] == 18 * (16 ** 3)
    
    # Microsecond logical clock cycle: tau_cycle = 12 * L * tau_rep = 12 * 16 * 13.8 ns = 2.6496 us
    assert 2.0 < res_16["logical_clock_cycle_us"] < 3.0
    
    # Sub-threshold logical error rate
    p_l = SFFCCLatticeArchitect.estimate_logical_error_rate(code_distance_l=16, physical_error_rate=0.024, threshold=0.08)
    assert p_l < 1e-6


def test_sparrow_fault_tolerance_referee():
    # 1. State-of-the-art compliant hardware spec
    good_spec = SparrowHardwareSpec(
        eta_end_to_end=0.93,          # Loss 7.0% < threshold 8.0%
        cyclicity_c=650.0,            # C > 574
        hom_visibility_ee=0.97,       # V_ee 97% > 96%
        hom_visibility_se=0.996,      # V_se 99.6% > 99.4%
        kappa_bar=8.0e-4,             # kappa < 1.2e-3
        t2_spin=113.0e-6,             # T_2 >> 417 * 13.8ns (5.75 us)
        t2_star=2.0e-6,               # T_2* >> 56 * 27.6ns (1.55 us)
        blinking_pa_pd_ratio=50.0,    # Ratio > 15.4
    )
    referee_good = SparrowFaultToleranceReferee(good_spec)
    report_good = referee_good.audit_hardware_specification()

    assert report_good["all_channels_passed"] is True
    assert report_good["is_fault_tolerant"] is True
    assert report_good["normalized_noise_budget_load"] < 1.0

    # 2. Non-compliant hardware spec (excessive loss: 12% > 8% threshold)
    bad_spec = SparrowHardwareSpec(eta_end_to_end=0.88)
    referee_bad = SparrowFaultToleranceReferee(bad_spec)
    report_bad = referee_bad.audit_hardware_specification()

    assert report_bad["all_channels_passed"] is False
    assert report_bad["is_fault_tolerant"] is False
    assert report_bad["channels"]["general_loss"]["passed"] is False
    assert report_bad["channels"]["general_loss"]["load"] > 1.0
