#!/usr/bin/env python3
"""Price the three benchmark champions on sFFCC photonics, as a NON-SUBMISSION.

Status: audit-only

The three competition benchmarks are scored from a compiled ``kickmix`` artifact:

    I   invsqrt     score = Q * toffoli_equivalent
    II  DQI         score = R(eps_plus) * Q^2 * T_mean
    III DFTHC-SOS   score = lambda_eff * T_total * Q_total

None of those expressions contains a time, a photon count, a fusion budget or a
detector. They are functions of the logical circuit alone. So re-hosting the same
logical circuit on a photonic machine leaves every score exactly where it was,
and a photonic circuit that is not a ``.kmb``/``.kmx`` artifact cannot be
submitted at all — the verifiers take the artifact, not a resource estimate.

This script writes that conclusion down as a machine-readable artifact: the
logical footprint of each champion, its sFFCC resource bill
(``apedp_dyn.quantum.sparrow_photonics``, arXiv:2507.16152), the wall-clock time
the photonic machine would need, and the explicit reason each row is *not* a
submission. It computes no score of its own and changes no receipt.

    python3 scripts/report_photonic_non_submission.py
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apedp_dyn.quantum.sparrow_photonics import SFFCCLatticeArchitect, SparrowHardwareSpec  # noqa: E402
OUT = REPO_ROOT / "artifacts/quantum_compilation/photonic_non_submission.json"

#: Operating point of the sFFCC blueprint used in its own worked example.
CODE_DISTANCE = 16


@dataclass(frozen=True)
class LogicalFootprint:
    """The only two quantities any of the three scores can see."""

    benchmark: str
    verifier: str
    qubits: int
    mean_ops: float
    ops_unit: str
    score_formula: str
    reported_score: float | None
    evidence: str

    @property
    def cycles(self) -> int:
        """One logical cycle per logical operation (the blueprint's own budget)."""
        return math.ceil(self.mean_ops)


CHAMPIONS = [
    LogicalFootprint(
        benchmark="I - invsqrt",
        verifier="verifiers/invsqrt/verify_submission_invsqrt.py",
        qubits=89,
        mean_ops=1770.7986610017997,
        ops_unit="Toffoli-equivalent (Toffolis + T/4)",
        score_formula="Q * toffoli_equivalent",
        reported_score=157601.08082916017,
        evidence="artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd_full_receipt.json",
    ),
    LogicalFootprint(
        benchmark="II - DQI",
        verifier="verifiers/dqi/verify_submission_dqi.py",
        qubits=1893,
        mean_ops=5_725_108.38,
        ops_unit="mean Toffoli-like per shot",
        score_formula="R(eps_plus) * Q^2 * T_mean",
        reported_score=None,  # R is a per-shot failure-rate penalty; not re-derived here.
        evidence="sealed Track 3 baseline receipt (see apedp_dyn/quantum/dqi_photonic_resource.py)",
    ),
    LogicalFootprint(
        benchmark="III - DFTHC-SOS",
        verifier="verifiers/dfthcsos/verify.py",
        qubits=730,
        mean_ops=93_237,
        ops_unit="T_total (Toffoli + T cost of the word circuits)",
        score_formula="lambda_eff * T_total * Q_total",
        reported_score=3_355_599_588.3155365,
        evidence="verifiers/dfthcsos/SCORING_README.md worked example",
    ),
]


def price(footprint: LogicalFootprint, lattice: dict[str, float], hardware: SparrowHardwareSpec) -> dict:
    """Resource bill for one shot of ``footprint`` on an sFFCC lattice."""
    cycles = footprint.cycles
    fusion_probability = hardware.eta_end_to_end**2
    fusions_per_cycle = lattice["encoded_fusions_per_cycle"] * footprint.qubits
    fusions_per_shot = fusions_per_cycle * cycles
    return {
        "benchmark": footprint.benchmark,
        "submission_status": "NON_SUBMISSION",
        "logical": {
            "qubits": footprint.qubits,
            "mean_ops": footprint.mean_ops,
            "ops_unit": footprint.ops_unit,
            "logical_cycles": cycles,
            "score_formula": footprint.score_formula,
            "reported_score": footprint.reported_score,
            "evidence": footprint.evidence,
        },
        "photonic": {
            "model": "sFFCC time-bin lattice (arXiv:2507.16152)",
            "code_distance": int(lattice["code_distance_L"]),
            "logical_clock_cycle_us": lattice["logical_clock_cycle_us"],
            "wall_clock_seconds_per_shot": cycles * lattice["logical_clock_cycle_sec"],
            "qd_emitters": int(lattice["num_qd_emitters"] * footprint.qubits),
            "photodetectors": int(lattice["num_photodetectors"] * footprint.qubits),
            "active_phase_shifters": int(lattice["active_phase_shifters_with_ebf"] * footprint.qubits),
            "passive_beamsplitters": int(lattice["passive_beamsplitters_with_ebf"] * footprint.qubits),
            "encoded_fusions_per_shot": int(fusions_per_shot),
            "fusion_success_probability": fusion_probability,
            "fusion_attempts_per_shot": fusions_per_shot / fusion_probability,
        },
        "score_under_platform_swap": footprint.reported_score,
        "score_delta": 0.0,
        "cannot_enter_benchmark": [
            "the verifiers consume a compiled .kmb/.kmx artifact; a resource estimate is not an artifact",
            f"the score {footprint.score_formula} contains no time, photon, fusion or detector term",
            "the logical circuit is unchanged, so Q and T are unchanged and the score is unchanged",
            "fault-tolerant photonics adds physical overhead that the score cannot see",
        ],
    }


def main() -> None:
    hardware = SparrowHardwareSpec()
    lattice = SFFCCLatticeArchitect.calculate_resources(CODE_DISTANCE, hardware)
    rows = [price(footprint, lattice, hardware) for footprint in CHAMPIONS]

    payload = {
        "artifact": "photonic_non_submission",
        "submission_status": "NON_SUBMISSION",
        "why": (
            "All three benchmark scores are functions of the logical circuit only. "
            "Re-hosting a circuit on photonic hardware cannot change them, and a "
            "photonic design cannot be submitted because the verifiers score a "
            "compiled kickmix artifact."
        ),
        "hardware_spec": asdict(hardware),
        "lattice": {k: v for k, v in lattice.items()},
        "benchmarks": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"sFFCC L={CODE_DISTANCE}: {lattice['logical_clock_cycle_us']:.4f} us per logical cycle\n")
    header = f"{'benchmark':18s} {'Q':>6s} {'T':>12s} {'score':>18s} {'wall/shot':>11s} {'emitters':>12s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        score = row["logical"]["reported_score"]
        score_text = f"{score:,.1f}" if score is not None else "R * Q^2 * T"
        print(
            f"{row['benchmark']:18s} {row['logical']['qubits']:6d} {row['logical']['mean_ops']:12,.1f} "
            f"{score_text:>18s} {row['photonic']['wall_clock_seconds_per_shot']:10.3f}s "
            f"{row['photonic']['qd_emitters']:12,d}"
        )
    print(f"\nEvery row is a NON_SUBMISSION with score_delta = 0.0. Wrote {OUT}")


if __name__ == "__main__":
    main()
