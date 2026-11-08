r"""DQI (Benchmark II) workload mapped onto the sFFCC deterministic-photonics model.

Status: experimental

Purpose
-------
Map the *reversible arithmetic* of the DQI Reed-Solomon syndrome decoder
(``verifiers/dqi/dqi_reference_circuit.py``: synchronized Extended Euclidean
Algorithm over GF(2^12), Chien search, Forney evaluation, and their uncompute
passes) onto the hardware-resource formulas already implemented in
``apedp_dyn.quantum.sparrow_photonics`` (sFFCC lattice, arXiv:2507.16152).

This module is a **resource accountant**, not a circuit compiler and not a
verifier. It answers one narrow question:

    If the existing DQI logical circuit were executed on an sFFCC time-bin
    photonic lattice of code distance L, how many quantum-dot emitters,
    photodetectors, active elements, and encoded fusions would be required,
    and how long would one shot take in wall-clock time?

Why this is worth doing at all
------------------------------
The Benchmark II score is ``R(eps_plus) * Q^2 * T_mean``. That expression has
no timing term. Consequently a photonic platform **cannot** improve the
competition score unless it changes ``Q`` or ``T_mean``; it can only change
wall-clock time and hardware composition. Recording that explicitly is the
main falsifiable finding this module produces, and it removes the most common
misreading of "photonic advantage" for this benchmark.

Known limitations (disclosed, not hidden)
-----------------------------------------
1. One logical sFFCC cycle is assumed to execute one Toffoli-like logical
   operation (``logical_ops_per_cycle``). The blueprint does not certify a
   per-gate cycle budget, so this is an assumption, not a measurement.
2. Fusion success is taken as ``eta^2`` (both photons survive to detection).
   Loss correction / fusion recycling overhead beyond the naive retry factor
   ``1 / eta^2`` is not modelled.
3. The workload's ``Q`` and ``T_mean`` are inputs. For the reference workload
   they come from the sealed Track 3 receipt; for the Gluzdov V6 workload they
   are an *analytical target* with no compiled circuit behind them.
4. No reliability claim is made about whether the DQI arithmetic (measurement-
   based uncomputation, register-cleanliness invariants) survives this mapping.
   That would require a compiled circuit plus the official verifier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from apedp_dyn.quantum.sparrow_photonics import (
    SFFCCLatticeArchitect,
    SparrowFaultToleranceReferee,
    SparrowFaultToleranceThresholds,
    SparrowHardwareSpec,
)

#: Code distance used as the default operating point in the Sparrow blueprint
#: (further-term noise target: p_L <= 1e-6). Not a competition threshold.
DEFAULT_CODE_DISTANCE: int = 16

#: Logical error rate the blueprint uses when quoting the L=16 operating point.
REFERENCE_LOGICAL_ERROR_RATE: float = 1e-6


@dataclass(frozen=True)
class DQIWorkload:
    """Logical resource footprint of one DQI decoder shot.

    ``Q`` and ``T_mean`` are exactly the two quantities that enter the official
    Benchmark II score, so carrying them as inputs keeps this module honest: it
    never invents a gain, it only re-prices the same circuit.
    """

    label: str
    qubits: int
    mean_toffoli_like: float
    eea_iterations: int = 222
    field_degree: int = 12
    syndromes: int = 70
    codeword_positions: int = 4095
    evidence: str = "sealed_baseline_receipt"

    def __post_init__(self) -> None:
        if self.qubits <= 0:
            raise ValueError("qubits must be positive")
        if self.mean_toffoli_like <= 0:
            raise ValueError("mean_toffoli_like must be positive")
        if self.eea_iterations <= 0:
            raise ValueError("eea_iterations must be positive")

    @property
    def toffoli_like_per_iteration(self) -> float:
        """Mean Toffoli-like operations per synchronized EEA iteration."""
        return self.mean_toffoli_like / self.eea_iterations

    @property
    def has_compiled_receipt(self) -> bool:
        return self.evidence == "sealed_baseline_receipt"

    @property
    def gf2_word_width(self) -> int:
        """Bits per GF(2^b) register word; one time-bin qubit per bit."""
        return self.field_degree


#: Reference Proos-Zalka decoder row from the sealed Track 3 baseline receipt.
DQI_REFERENCE_WORKLOAD = DQIWorkload(
    label="proos_zalka_reference_n_iters_222",
    qubits=1893,
    mean_toffoli_like=5_725_108.38,
    eea_iterations=222,
    evidence="sealed_baseline_receipt",
)

#: Gluzdov V6 degree-register-recycling row. These numbers are a *target*:
#: no compiled .kmb circuit has produced them, so this row must never be
#: treated as a verified result.
DQI_GLUZDOV_V6_TARGET_WORKLOAD = DQIWorkload(
    label="gluzdov_v6_degree_register_recycling_target",
    qubits=1789,
    mean_toffoli_like=445_229.0,
    eea_iterations=222,
    evidence="analytical_target_no_receipt",
)


@dataclass(frozen=True)
class DQIPhotonicMappingAssumptions:
    """Assumptions required to convert logical operations into lattice cycles."""

    code_distance: int = DEFAULT_CODE_DISTANCE
    logical_ops_per_cycle: float = 1.0
    fusion_success_probability: float | None = None
    basis: str = (
        "one sFFCC logical cycle per Toffoli-like logical operation; "
        "fusion success = eta^2; retries = 1/fusion_success"
    )

    def __post_init__(self) -> None:
        if self.code_distance < 3:
            raise ValueError("code_distance must be at least 3")
        if self.logical_ops_per_cycle <= 0:
            raise ValueError("logical_ops_per_cycle must be positive")
        p = self.fusion_success_probability
        if p is not None and not (0.0 < p <= 1.0):
            raise ValueError("fusion_success_probability must lie in (0, 1]")


@dataclass(frozen=True)
class SuperconductingBaseline:
    """Caller-supplied superconducting reference point for a wall-clock comparison.

    Deliberately has **no defaults**. The two numbers below are assumptions about
    a platform this repository does not model, so the caller has to state them
    explicitly; a default would quietly launder an assumption into a "baseline".

    ``logical_cycle_seconds`` is the surface-code logical cycle (one round of
    syndrome extraction and decoding). ``physical_qubits_per_logical`` is the
    space overhead assumed per logical qubit, including routing and factory
    ancillas.
    """

    logical_cycle_seconds: float
    physical_qubits_per_logical: float
    label: str = "caller_supplied_superconducting_baseline"

    def __post_init__(self) -> None:
        if self.logical_cycle_seconds <= 0:
            raise ValueError("logical_cycle_seconds must be positive")
        if self.physical_qubits_per_logical <= 0:
            raise ValueError("physical_qubits_per_logical must be positive")


class DQIPhotonicResourceMapper:
    """Prices a DQI logical workload against the sFFCC photonic lattice model."""

    def __init__(
        self,
        hardware: SparrowHardwareSpec | None = None,
        assumptions: DQIPhotonicMappingAssumptions | None = None,
        thresholds: SparrowFaultToleranceThresholds | None = None,
    ) -> None:
        self.hardware = hardware or SparrowHardwareSpec()
        self.assumptions = assumptions or DQIPhotonicMappingAssumptions()
        self.thresholds = thresholds or SparrowFaultToleranceThresholds()

    # -- derived hardware quantities -------------------------------------

    @property
    def fusion_success_probability(self) -> float:
        """Naive Type-II fusion success: both photons survive to detection."""
        if self.assumptions.fusion_success_probability is not None:
            return self.assumptions.fusion_success_probability
        return self.hardware.eta_end_to_end ** 2

    def fault_tolerance_audit(self) -> dict[str, Any]:
        """Run the existing 8-channel Table 2 referee on the configured hardware."""
        return SparrowFaultToleranceReferee(
            self.hardware, self.thresholds
        ).audit_hardware_specification()

    # -- the mapping ------------------------------------------------------

    def map_workload(
        self,
        workload: DQIWorkload,
        *,
        r_overhead: float | None = None,
    ) -> dict[str, Any]:
        """Return the photonic resource bill for one shot of ``workload``.

        ``r_overhead`` is the decoder's ``R(eps_plus)`` failure-rate penalty. When
        supplied, the platform-invariance check recomputes the official score
        ``R * Q^2 * T_mean`` and shows it is identical for every code distance.
        """
        a = self.assumptions
        lattice = SFFCCLatticeArchitect.calculate_resources(a.code_distance, self.hardware)

        emitters_per_patch = lattice["num_qd_emitters"]
        detectors_per_patch = lattice["num_photodetectors"]
        active_per_patch = lattice["active_phase_shifters_with_ebf"]
        fusions_per_cycle_per_patch = lattice["encoded_fusions_per_cycle"]
        cycle_seconds = lattice["logical_clock_cycle_sec"]

        cycles_per_shot = math.ceil(workload.mean_toffoli_like / a.logical_ops_per_cycle)
        fusions_per_cycle = fusions_per_cycle_per_patch * workload.qubits
        fusions_per_shot = fusions_per_cycle * cycles_per_shot
        p_fusion = self.fusion_success_probability
        fusion_attempts_per_shot = fusions_per_shot / p_fusion

        wall_clock_seconds = cycles_per_shot * cycle_seconds
        cycles_per_iteration = cycles_per_shot / workload.eea_iterations

        # GF(2^b) word accounting: every register word is `field_degree` time-bin
        # qubits, and the EEA's cost is concentrated in per-iteration division
        # and comparison passes over the m = n + 3 polynomial registers.
        m_registers = workload.syndromes + 3
        per_iteration = workload.toffoli_like_per_iteration

        ft = self.fault_tolerance_audit()
        logical_error_rate = SFFCCLatticeArchitect.estimate_logical_error_rate(
            a.code_distance, operating_regime="further_term"
        )

        result: dict[str, Any] = {
            "status": "ANALYTICAL_PROJECTION_ONLY",
            "compiled_kmb": False,
            "official_verifier_run": False,
            "workload": {
                "label": workload.label,
                "evidence": workload.evidence,
                "has_compiled_receipt": workload.has_compiled_receipt,
                "qubits": workload.qubits,
                "mean_toffoli_like": workload.mean_toffoli_like,
                "eea_iterations": workload.eea_iterations,
                "field_degree": workload.field_degree,
                "syndromes": workload.syndromes,
                "codeword_positions": workload.codeword_positions,
            },
            "per_patch_resources": {
                "code_distance_L": lattice["code_distance_L"],
                "emitters": emitters_per_patch,
                "photodetectors": detectors_per_patch,
                "active_elements_with_ebf": active_per_patch,
                "active_elements_saved_by_ebf": lattice["active_elements_saved_by_ebf"],
                "encoded_fusions_per_cycle": fusions_per_cycle_per_patch,
            },
            "hardware_totals": {
                "logical_patches": workload.qubits,
                "emitters": emitters_per_patch * workload.qubits,
                "photodetectors": detectors_per_patch * workload.qubits,
                "active_elements": active_per_patch * workload.qubits,
                "fusions_per_cycle": fusions_per_cycle,
                "cycles_per_shot": cycles_per_shot,
                "fusions_per_shot": fusions_per_shot,
                "fusion_success_probability": p_fusion,
                "expected_fusion_attempts_per_shot": fusion_attempts_per_shot,
            },
            "timing": {
                "logical_cycle_us": lattice["logical_clock_cycle_us"],
                "logical_clock_khz": lattice["clock_frequency_khz"],
                "cycles_per_shot": cycles_per_shot,
                "cycles_per_eea_iteration": cycles_per_iteration,
                "wall_clock_seconds_per_shot": wall_clock_seconds,
                "wall_clock_hours_per_shot": wall_clock_seconds / 3600.0,
            },
            "eea_arithmetic_mapping": {
                "gf2_word_bits": workload.gf2_word_width,
                "polynomial_registers": m_registers,
                "toffoli_like_per_iteration": per_iteration,
                "toffoli_like_per_register_per_iteration": per_iteration / m_registers,
                "iterations": workload.eea_iterations,
                "total_toffoli_like": workload.mean_toffoli_like,
                "uncompute_pass_mirrors_forward_pass": True,
            },
            "fault_tolerance": {
                "is_fault_tolerant": ft["is_fault_tolerant"],
                "normalized_noise_budget_load": ft["normalized_noise_budget_load"],
                "all_channels_passed": ft["all_channels_passed"],
                "logical_error_rate_at_L": logical_error_rate,
                "reference_logical_error_rate": REFERENCE_LOGICAL_ERROR_RATE,
                "meets_reference_logical_error_rate": logical_error_rate
                <= REFERENCE_LOGICAL_ERROR_RATE,
            },
            "score_invariance": {
                "scoring_formula": "R(eps_plus) * Q^2 * T_mean",
                "contains_timing_term": False,
                "platform_invariant_given_same_Q_and_T": True,
            },
            "assumptions": {
                "code_distance_L": a.code_distance,
                "logical_ops_per_cycle": a.logical_ops_per_cycle,
                "basis": a.basis,
            },
            "promotion_decision": "HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT",
        }

        if r_overhead is not None:
            if r_overhead <= 0:
                raise ValueError("r_overhead must be positive")
            score = r_overhead * (workload.qubits ** 2) * workload.mean_toffoli_like
            result["score_invariance"]["r_overhead"] = r_overhead
            result["score_invariance"]["official_score"] = score
            result["score_invariance"]["note"] = (
                "This score is identical at every code distance because no optical "
                "timing enters Q^2 * T_mean. Photonics changes wall-clock and hardware "
                "composition, not the competition score, unless Q or T_mean changes."
            )

        return result

    def compare_workloads(
        self,
        reference: DQIWorkload,
        candidate: DQIWorkload,
        *,
        r_overhead: float | None = None,
    ) -> dict[str, Any]:
        """Diff two workloads on both the official score and the photonic bill."""
        ref = self.map_workload(reference, r_overhead=r_overhead)
        cand = self.map_workload(candidate, r_overhead=r_overhead)

        qubit_ratio = candidate.qubits / reference.qubits
        toffoli_ratio = candidate.mean_toffoli_like / reference.mean_toffoli_like

        return {
            "status": "ANALYTICAL_PROJECTION_ONLY",
            "reference_label": reference.label,
            "candidate_label": candidate.label,
            "candidate_has_compiled_receipt": candidate.has_compiled_receipt,
            "qubit_ratio": qubit_ratio,
            "toffoli_ratio": toffoli_ratio,
            "logical_cost_ratio": (qubit_ratio ** 2) * toffoli_ratio,
            "projected_gain_factor": 1.0 / ((qubit_ratio ** 2) * toffoli_ratio),
            "emitter_ratio": (
                cand["hardware_totals"]["emitters"] / ref["hardware_totals"]["emitters"]
            ),
            "wall_clock_ratio": (
                cand["timing"]["wall_clock_seconds_per_shot"]
                / ref["timing"]["wall_clock_seconds_per_shot"]
            ),
            "reasons_to_distrust": [
                item
                for item, flagged in (
                    (
                        "candidate workload is an analytical target with no compiled circuit",
                        not candidate.has_compiled_receipt,
                    ),
                    (
                        "no official verifier receipt exists for the candidate row",
                        not candidate.has_compiled_receipt,
                    ),
                )
                if flagged
            ],
        }


    def compare_platforms(
        self,
        workload: DQIWorkload,
        superconducting: SuperconductingBaseline,
        *,
        r_overhead: float | None = None,
    ) -> dict[str, Any]:
        """Compare wall-clock and hardware composition against a stated baseline.

        The official score is not compared here because it cannot differ: the
        same ``Q`` and ``T_mean`` enter it regardless of platform. This method
        answers only the two questions a platform actually changes — how long a
        shot takes, and what hardware has to exist to run it.
        """
        photonic = self.map_workload(workload, r_overhead=r_overhead)
        cycles = photonic["timing"]["cycles_per_shot"]

        sc_seconds = cycles * superconducting.logical_cycle_seconds
        sc_physical_qubits = workload.qubits * superconducting.physical_qubits_per_logical
        photonic_devices = (
            photonic["hardware_totals"]["emitters"]
            + photonic["hardware_totals"]["photodetectors"]
            + photonic["hardware_totals"]["active_elements"]
        )
        ph_seconds = photonic["timing"]["wall_clock_seconds_per_shot"]

        return {
            "status": "ANALYTICAL_PROJECTION_ONLY",
            "official_verifier_run": False,
            "workload_label": workload.label,
            "assumptions_disclosure": {
                "photonic": self.assumptions.basis,
                "superconducting": (
                    "caller-supplied cycle time and per-logical-qubit space overhead; "
                    "this repository does not model a superconducting machine"
                ),
                "superconducting_label": superconducting.label,
            },
            "wall_clock": {
                "logical_cycles": cycles,
                "photonic_seconds_per_shot": ph_seconds,
                "superconducting_seconds_per_shot": sc_seconds,
                "photonic_over_superconducting": ph_seconds / sc_seconds,
                "photonic_cycle_us": photonic["timing"]["logical_cycle_us"],
                "superconducting_cycle_us": superconducting.logical_cycle_seconds * 1e6,
            },
            "hardware_composition": {
                "logical_qubits": workload.qubits,
                "photonic_emitters": photonic["hardware_totals"]["emitters"],
                "photonic_photodetectors": photonic["hardware_totals"]["photodetectors"],
                "photonic_active_elements": photonic["hardware_totals"]["active_elements"],
                "photonic_devices_total": photonic_devices,
                "superconducting_physical_qubits": sc_physical_qubits,
                "superconducting_control_lines_lower_bound": sc_physical_qubits,
                "devices_per_logical_qubit_photonic": photonic_devices / workload.qubits,
                "devices_per_logical_qubit_superconducting": (
                    sc_physical_qubits / workload.qubits
                ),
            },
            "score_comparison": {
                "applicable": False,
                "reason": (
                    "R(eps_plus) * Q^2 * T_mean is platform-independent for a fixed "
                    "circuit, so a platform swap cannot change it"
                ),
            },
        }


def map_reference_workload(*, code_distance: int = DEFAULT_CODE_DISTANCE) -> dict[str, Any]:
    """Convenience wrapper: price the sealed reference DQI workload."""
    mapper = DQIPhotonicResourceMapper(
        assumptions=DQIPhotonicMappingAssumptions(code_distance=code_distance)
    )
    return mapper.map_workload(DQI_REFERENCE_WORKLOAD)


__all__ = [
    "DEFAULT_CODE_DISTANCE",
    "DQI_GLUZDOV_V6_TARGET_WORKLOAD",
    "DQI_REFERENCE_WORKLOAD",
    "REFERENCE_LOGICAL_ERROR_RATE",
    "DQIPhotonicMappingAssumptions",
    "DQIPhotonicResourceMapper",
    "DQIWorkload",
    "SuperconductingBaseline",
    "map_reference_workload",
]
