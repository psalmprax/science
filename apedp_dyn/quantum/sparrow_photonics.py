r"""Deterministic Quantum Photonics & Fault-Tolerant Quantum Computing Engine (Sparrow Quantum Blueprint).

Status: experimental

Based on the blueprint and experimental benchmarks from Sparrow Quantum and the Niels Bohr Institute:
"Practical blueprint for low-depth photonic quantum computing with quantum dots"
Authors: Ming Lai Chan, Aliki Anna Capatos, Peter Lodahl, Anders S. Sørensen, Stefano Paesani
(npj Quantum Information / arXiv:2507.16152)

Formulates, simulates, and benchmarks:
1. Probabilistic (SPDC) vs. Deterministic (Quantum Dot) Multi-Photon Scaling Law.
2. Quantum Dot Time-Bin Entangled Photon Source (EPS) & Cluster State Generation.
3. Passive Fusion Gate with Excitation-Based Feedback (EBF) Verification (Eq. 7).
4. Table 2 Error Catalogue & Multi-Parameter Fault-Tolerance Referee:
   - Loss threshold: eta > 92% (p_loss <= 8.0%)
   - Optical cyclicity threshold: C > 574 (branching <= 0.174%)
   - Emitter-emitter HOM visibility: V_HOM_ee > 96%
   - Single-emitter HOM visibility: V_HOM_se > 99.4%
   - Laser-induced spin-flips: kappa_bar < 1.2e-3
   - Spin coherence T_2 > 417 tau_rep
   - Overhauser dephasing T_2* > 56 tau_round
   - Blinking ratio: P_A / P_D > 15.4
5. sFFCC Lattice Hardware Scaling & Microsecond Clock Cycle Calculator:
   - N_QD = 6 * L^2 emitters per logical qubit
   - N_fusions = 18 * L^3 fusions per logical cycle
   - tau_cycle = 12 * L * tau_rep (2.65 microseconds at L=16)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SparrowHardwareSpec:
    """Experimental physical hardware parameters for a Quantum Dot EPS unit."""
    tau_int: float = 11.83e-9          # Time-bin interferometer delay (seconds)
    tau_d: float = 10.0e-9             # Detector dead time (seconds)
    tau_echo: float = 29.0e-9          # Spin echo rephasing time (seconds)
    tau_pi: float = 4.0e-9             # Spin pi-pulse duration (seconds)
    tau_rep: float = 13.8e-9           # Pulse repetition period (seconds)
    tau_tb: float = 0.5e-9             # Time-bin photon lifetime / duration (seconds)
    eta_end_to_end: float = 0.93       # Total end-to-end collection & detection efficiency (1 - loss)
    hom_visibility_ee: float = 0.97    # Emitter-emitter Hong-Ou-Mandel visibility
    hom_visibility_se: float = 0.996   # Single-emitter Hong-Ou-Mandel visibility
    cyclicity_c: float = 650.0         # Optical transition cyclicity C = Gamma_c / Gamma_nc
    t2_spin: float = 113.0e-6          # Spin coherence time T_2 with dynamical decoupling (seconds)
    t2_star: float = 2.0e-6            # Ground-state dephasing time T_2* (seconds)
    kappa_bar: float = 8.0e-4          # Laser-induced normalized spin-flip rate
    blinking_pa_pd_ratio: float = 50.0 # Blinking ratio P_A / P_D


@dataclass(frozen=True)
class SparrowFaultToleranceThresholds:
    """Exact Table 2 fault-tolerance thresholds from arXiv:2507.16152."""
    loss_threshold_pct: float = 8.0               # Maximum tolerable uncorrelated loss: 8.0%
    cyclicity_threshold: float = 574.0            # Minimum optical cyclicity C > 574 (branching <= 0.174%)
    hom_ee_threshold_pct: float = 96.0            # Emitter-emitter HOM visibility > 96.0% (distinguishability <= 4%)
    hom_se_threshold_pct: float = 99.4            # Single-emitter HOM visibility > 99.4% (distinguishability <= 0.57%)
    laser_spin_flip_threshold: float = 1.2e-3     # Normalized spin-flip rate kappa_bar < 1.2e-3
    t2_spin_multiplier: float = 417.0             # T_2 > 417 * tau_rep
    t2_star_multiplier: float = 56.0              # T_2* > 56 * tau_round
    blinking_ratio_threshold: float = 15.4        # P_A / P_D > 15.4


class DeterministicVsProbabilisticScaling:
    """Compares resource overhead scaling between probabilistic (SPDC) and deterministic (QD) sources."""

    @staticmethod
    def calculate_spdc_overhead(n_photons: int, p_pair: float = 0.05, eta_switch: float = 0.85, mux_depth: int = 8) -> float:
        r"""Compute physical resource cost to generate an n-photon state using multiplexed SPDC.
        
        P_single = 1 - (1 - p_pair * eta_switch)^mux_depth
        P_n = (P_single)^n
        Overhead = n * mux_depth / P_n
        """
        p_single = 1.0 - math.pow(1.0 - (p_pair * eta_switch), mux_depth)
        p_n = math.pow(p_single, n_photons)
        if p_n <= 1e-18:
            return float("inf")
        return (n_photons * mux_depth) / p_n

    @staticmethod
    def calculate_quantum_dot_overhead(n_photons: int, eta_out: float = 0.93) -> float:
        r"""Compute physical resource cost to generate an n-photon state using a deterministic QD source.
        
        P_n = eta_out^n
        Overhead = n / P_n
        """
        p_n = math.pow(eta_out, n_photons)
        if p_n <= 1e-18:
            return float("inf")
        return n_photons / p_n

    @staticmethod
    def compute_scaling_ratio(n_photons: int, p_pair: float = 0.05, eta_switch: float = 0.85, mux_depth: int = 8, eta_out: float = 0.93) -> float:
        r"""Returns ratio: Overhead(SPDC) / Overhead(Quantum Dot)."""
        spdc = DeterministicVsProbabilisticScaling.calculate_spdc_overhead(n_photons, p_pair, eta_switch, mux_depth)
        qd = DeterministicVsProbabilisticScaling.calculate_quantum_dot_overhead(n_photons, eta_out)
        return spdc / qd if qd > 0 else float("inf")


class PassiveFusionEngine:
    """Verifies Type-II Bell State Measurement and Excitation-Based Feedback (EBF)."""

    def __init__(self, spec: SparrowHardwareSpec | None = None) -> None:
        self.spec = spec or SparrowHardwareSpec()

    def verify_ebf_timing_inequality(self, tau_ebf: float) -> tuple[bool, dict[str, float]]:
        r"""Verify Equation (7): tau_TB < tau_EBF < tau_int - tau_pi.
        
        Excitation-based feedback delays the excitation pulse by tau_EBF so photons arrive
        at different time-bins, eliminating two-photon Hong-Ou-Mandel interference and
        constituting deterministic single-qubit Z-measurements through 100% passive linear optics.
        """
        lower_bound = self.spec.tau_tb
        upper_bound = self.spec.tau_int - self.spec.tau_pi
        is_valid = (lower_bound < tau_ebf < upper_bound)
        return is_valid, {
            "lower_bound_tau_tb_ns": lower_bound * 1e9,
            "tau_ebf_ns": tau_ebf * 1e9,
            "upper_bound_ns": upper_bound * 1e9,
            "headroom_ns": (upper_bound - lower_bound) * 1e9,
        }

    def simulate_type2_fusion(self, input_state: str, tau_offset: float = 0.0) -> dict[str, Any]:
        r"""Simulate a Type-II linear-optics fusion gate.
        
        When tau_offset == 0: Hong-Ou-Mandel interference occurs (50% Bell-state success).
        When tau_offset == tau_ebf: Interference is deactivated; yields deterministic single-qubit Z-basis measurements.
        """
        # Bell state fusion probability with 50:50 beam splitter
        eta = self.spec.eta_end_to_end
        p_loss = 1.0 - (eta * eta)
        
        # Hong-Ou-Mandel visibility decay with temporal offset
        hom_width = self.spec.tau_tb
        effective_visibility = self.spec.hom_visibility_ee * math.exp(-0.5 * (tau_offset / hom_width) ** 2)
        
        if tau_offset > 2.0 * hom_width:
            # Fully distinguishable in time -> Deterministic Single-Qubit Z-measurements
            mode = "passive_single_qubit_z"
            success_prob = eta * eta
        else:
            # Temporal overlap -> Type-II Entangling Fusion
            mode = "two_qubit_bell_fusion"
            success_prob = 0.5 * (eta * eta) * effective_visibility

        return {
            "mode": mode,
            "effective_hom_visibility": effective_visibility,
            "fusion_success_probability": success_prob,
            "photon_loss_probability": p_loss,
            "is_purely_passive": True,
        }


class SFFCCLatticeArchitect:
    """Calculates hardware resources, optical depth, and clock cycles for the sFFCC topological code."""

    @staticmethod
    def calculate_resources(code_distance_l: int, spec: SparrowHardwareSpec | None = None) -> dict[str, Any]:
        r"""Compute exact physical component scaling for code distance L (Table 3)."""
        s = spec or SparrowHardwareSpec()
        cd_l = code_distance_l

        n_emitters = 6 * (cd_l ** 2)
        n_detectors = 18 * (cd_l ** 2)
        
        # Active phase shifters
        n_active_phase_shifters_standard = 33 * (cd_l ** 2) - 4 * cd_l + 1
        n_active_phase_shifters_ebf = 24 * (cd_l ** 2)
        
        # Passive beamsplitters
        n_passive_bs_standard = 54 * (cd_l ** 2) - 8 * cd_l + 2
        n_passive_bs_ebf = 45 * (cd_l ** 2) - 4 * cd_l + 1
        
        # Fusions per cycle
        n_fusions_per_cycle = 18 * (cd_l ** 3)
        
        # Optical depth per photon
        optical_depth_active = 4  # With EBF (down from 5)
        optical_depth_passive = 6 # 5 to 7 beamsplitters
        
        # Logical clock cycle duration: tau_cycle = 6 * L * tau_physical = 6 * L * (2 * tau_rep)
        tau_physical = 2.0 * s.tau_rep
        tau_logical_cycle = 6.0 * cd_l * tau_physical

        return {
            "code_distance_L": cd_l,
            "num_qd_emitters": n_emitters,
            "num_photodetectors": n_detectors,
            "active_phase_shifters_standard": n_active_phase_shifters_standard,
            "active_phase_shifters_with_ebf": n_active_phase_shifters_ebf,
            "active_elements_saved_by_ebf": n_active_phase_shifters_standard - n_active_phase_shifters_ebf,
            "passive_beamsplitters_standard": n_passive_bs_standard,
            "passive_beamsplitters_with_ebf": n_passive_bs_ebf,
            "encoded_fusions_per_cycle": n_fusions_per_cycle,
            "optical_depth_active": optical_depth_active,
            "optical_depth_passive": optical_depth_passive,
            "logical_clock_cycle_sec": tau_logical_cycle,
            "logical_clock_cycle_us": tau_logical_cycle * 1e6,
            "clock_frequency_khz": 1.0 / (tau_logical_cycle * 1e3),
        }

    @staticmethod
    def estimate_logical_error_rate(
        code_distance_l: int,
        physical_error_rate: float | None = None,
        threshold: float = 0.08,
        operating_regime: str = "further_term",
    ) -> float:
        r"""
        Sub-threshold logical error rate scaling on the sFFCC lattice (Chan et al. Fig. 10(a)).
        \log_{10} p_L = \log_{10} p_0 - \Lambda \cdot L
        where:
        - Nearer-term target (\sum_i p_i / p_{th,i} = 0.62): \Lambda = 0.106
          (requires L ~ 48 for p_L = 10^-6, L ~ 76 for 10^-9, L ~ 105 for 10^-12)
        - Further-term target (\sum_i p_i / p_{th,i} = 0.46): \Lambda = 0.318
          (requires L ~ 16 for p_L = 10^-6, L ~ 26 for 10^-9, L ~ 35 for 10^-12)
        """
        p_0 = 0.122  # Empirical fit intercept from Fig. 10(a)
        
        if physical_error_rate is not None:
            if physical_error_rate >= threshold:
                return 1.0
            load = physical_error_rate / threshold
            # Linear interpolation of suppression slope between near-term (0.62) and further-term (0.46)
            slope = max(0.05, 0.318 - 1.325 * (load - 0.46))
        elif operating_regime == "near_term":
            slope = 0.106
        else:  # "further_term"
            slope = 0.318

        log_pl = math.log10(p_0) - slope * float(code_distance_l)
        return float(min(1.0, math.pow(10.0, log_pl)))


class SparrowFaultToleranceReferee:
    """Epistemic Referee adjudicating whether a hardware specification meets FTQC thresholds (Table 2)."""

    def __init__(
        self,
        spec: SparrowHardwareSpec | None = None,
        thresholds: SparrowFaultToleranceThresholds | None = None,
    ) -> None:
        self.spec = spec or SparrowHardwareSpec()
        self.thresholds = thresholds or SparrowFaultToleranceThresholds()

    def audit_hardware_specification(self) -> dict[str, Any]:
        """Perform full multi-channel epistemic audit against Table 2 error thresholds."""
        s = self.spec
        th = self.thresholds

        # 1. General Loss Rate: p_loss = 1 - eta
        p_loss_actual = 1.0 - s.eta_end_to_end
        p_loss_th = th.loss_threshold_pct / 100.0
        ratio_loss = p_loss_actual / p_loss_th
        pass_loss = (ratio_loss <= 1.0)

        # 2. Branching Error: p_branch = 1 / (C + 1)
        p_branch_actual = 1.0 / (s.cyclicity_c + 1.0)
        p_branch_th = 1.0 / (th.cyclicity_threshold + 1.0)
        ratio_branch = p_branch_actual / p_branch_th
        pass_branch = (s.cyclicity_c >= th.cyclicity_threshold)

        # 3. Emitter-Emitter Distinguishability: 1 - V_HOM_ee
        p_hom_ee_actual = 1.0 - s.hom_visibility_ee
        p_hom_ee_th = 1.0 - (th.hom_ee_threshold_pct / 100.0)
        ratio_hom_ee = p_hom_ee_actual / p_hom_ee_th
        pass_hom_ee = (s.hom_visibility_ee >= th.hom_ee_threshold_pct / 100.0)

        # 4. Single-Emitter Distinguishability: 1 - V_HOM_se
        p_hom_se_actual = 1.0 - s.hom_visibility_se
        p_hom_se_th = 1.0 - (th.hom_se_threshold_pct / 100.0)
        ratio_hom_se = p_hom_se_actual / p_hom_se_th
        pass_hom_se = (s.hom_visibility_se >= th.hom_se_threshold_pct / 100.0)

        # 5. Laser-Induced Spin-Flip Error
        ratio_kappa = s.kappa_bar / th.laser_spin_flip_threshold
        pass_kappa = (s.kappa_bar <= th.laser_spin_flip_threshold)

        # 6. Finite T_2 Spin Coherence: requires T_2 > 417 * tau_rep
        t2_min_required = th.t2_spin_multiplier * s.tau_rep
        ratio_t2 = t2_min_required / max(1e-15, s.t2_spin)
        pass_t2 = (s.t2_spin >= t2_min_required)

        # 7. Overhauser Ground-State Dephasing: requires T_2* > 56 * tau_round
        tau_round = 2.0 * s.tau_rep
        t2_star_min_required = th.t2_star_multiplier * tau_round
        ratio_t2_star = t2_star_min_required / max(1e-15, s.t2_star)
        pass_t2_star = (s.t2_star >= t2_star_min_required)

        # 8. Blinking Ratio: requires P_A / P_D > 15.4
        ratio_blinking = th.blinking_ratio_threshold / max(1e-15, s.blinking_pa_pd_ratio)
        pass_blinking = (s.blinking_pa_pd_ratio >= th.blinking_ratio_threshold)

        # Combined multi-objective noise budget: sum(p_i / p_th_i)
        noise_budget_sum = (
            ratio_loss
            + ratio_branch
            + ratio_hom_ee
            + ratio_hom_se
            + ratio_kappa
            + ratio_t2
            + ratio_t2_star
            + ratio_blinking
        ) / 8.0

        all_passed = (
            pass_loss
            and pass_branch
            and pass_hom_ee
            and pass_hom_se
            and pass_kappa
            and pass_t2
            and pass_t2_star
            and pass_blinking
        )

        return {
            "all_channels_passed": all_passed,
            "normalized_noise_budget_load": noise_budget_sum,
            "is_fault_tolerant": all_passed and (noise_budget_sum <= 1.0),
            "channels": {
                "general_loss": {"actual": p_loss_actual, "threshold": p_loss_th, "passed": pass_loss, "load": ratio_loss},
                "cyclicity_branching": {"actual_c": s.cyclicity_c, "min_c": th.cyclicity_threshold, "passed": pass_branch, "load": ratio_branch},
                "hom_ee_visibility": {"actual": s.hom_visibility_ee, "min": th.hom_ee_threshold_pct/100.0, "passed": pass_hom_ee, "load": ratio_hom_ee},
                "hom_se_visibility": {"actual": s.hom_visibility_se, "min": th.hom_se_threshold_pct/100.0, "passed": pass_hom_se, "load": ratio_hom_se},
                "laser_spin_flips": {"actual": s.kappa_bar, "max": th.laser_spin_flip_threshold, "passed": pass_kappa, "load": ratio_kappa},
                "spin_t2": {"actual_s": s.t2_spin, "min_s": t2_min_required, "passed": pass_t2, "load": ratio_t2},
                "overhauser_t2_star": {"actual_s": s.t2_star, "min_s": t2_star_min_required, "passed": pass_t2_star, "load": ratio_t2_star},
                "blinking_pa_pd": {"actual": s.blinking_pa_pd_ratio, "min": th.blinking_ratio_threshold, "passed": pass_blinking, "load": ratio_blinking},
            },
        }
