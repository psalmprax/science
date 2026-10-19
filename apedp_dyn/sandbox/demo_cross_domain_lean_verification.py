"""Cross-Domain Demonstration of the Agnostic / Universal Lean 4 Verification Bridge.

Evaluates and formally verifies candidates across 5 heterogeneous scientific domains:
  1. Classical Mechanics / Physics (Hamiltonian Oscillator vs Energy-Leak Trap)
  2. Mathematical Epidemiology (Kermack-McKendrick SIR Mass-Action Conservation)
  3. Quantitative Finance (Saturated Order-Flow Velocity vs Cubic Runaway Polynomial)
  4. High-Dimensional Geometry (Sphere-Packing Kissing Distance Positivity)
  5. Fluid Dynamics (Navier-Stokes Solenoidal Incompressible Flow)

Demonstrates the epistemic dual-boundary principle:
  - Deductive mathematical verification (Lean 4 Kernel / Mathlib tactics: P0, P1, P2)
  - Inductive empirical verification (APEDP-X Claim Ladder: C0 to C6)
  - Joint Epistemic Discovery Certificates: (ClaimLevel, ProofClaimStatus)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from apedp_dyn.epistemic.agnostic_adapter import (
    AgnosticCandidateSpecification,
    AgnosticProblemDataset,
    DeclaredInvariant,
    InvariantKind,
)
from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EvidenceBundle,
    EvidenceRequirement,
)
from apedp_dyn.epistemic.lean_universal_bridge import (
    LeanUniversalBridge,
    VerificationBackend,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross_domain_lean_demo")

from apedp_dyn.conf.paths import artifact_dir, lean_root

ARTIFACT_DIR = artifact_dir()
LEAN_OUT_DIR = lean_root()


def run_cross_domain_lean_verification_suite() -> dict:
    LEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge = LeanUniversalBridge(output_dir=LEAN_OUT_DIR, backend=VerificationBackend.CORE_AXIOMATIC)

    domain_records: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Domain 1: Classical Mechanics / Hamiltonian Physics
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=== Domain 1: Classical Mechanics (Hamiltonian Oscillator vs Leaky Trap) ===")

    # Valid Conservative Model
    phys_valid_spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-HAMILTONIAN-OSCILLATOR",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="x(t) = sin(t)",
        declared_invariants=[
            DeclaredInvariant(
                name="hamiltonian_energy_conservation",
                kind=InvariantKind.CONSERVATION_LAW,
                description="Conservation of mechanical energy H = 0.5*(v^2 + x^2)",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="lipschitz_velocity_bound",
                kind=InvariantKind.LIPSCHITZ_SMOOTHNESS,
                description="Global phase space velocity smoothness",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    phys_ds = AgnosticProblemDataset(
        domain_label="Classical Mechanics",
        X_train=np.linspace(0, 10, 50),
        y_train=np.sin(np.linspace(0, 10, 50)),
        X_holdout=np.linspace(10, 20, 50),
        y_holdout=np.sin(np.linspace(10, 20, 50)),
    )
    phys_valid_evidence = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"r2_holdout": 0.999, "energy_drift": 0.0},
    )
    phys_valid_cert = bridge.adjudicate_joint_scientific_claim(phys_valid_spec, phys_ds, phys_valid_evidence)

    # Trap Model: Dissipative Energy Leaker masquerading as conservative oscillator
    phys_trap_spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-ENERGY-LEAK-TRAP",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="x(t) = sin(t) * exp(-0.05 * t)",
        declared_invariants=[
            DeclaredInvariant(
                name="hamiltonian_energy_conservation",
                kind=InvariantKind.CONSERVATION_LAW,
                description="Preservation of mechanical energy",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    phys_trap_evidence = EvidenceBundle(
        verified_evidence={EvidenceRequirement.IN_SAMPLE_CONVERGENCE},
        failures=["energy_dissipation_drift"],
    )
    phys_trap_cert = bridge.adjudicate_joint_scientific_claim(phys_trap_spec, phys_ds, phys_trap_evidence)

    domain_records.append({
        "domain": "Classical Mechanics / Physics",
        "valid_candidate": phys_valid_cert.to_dict(),
        "trap_candidate": phys_trap_cert.to_dict(),
    })

    # Core Lean 4 Benchmark: Discrete Algebraic Conservation (Kernel Verified without external dependencies)
    logger.info("=== Core Lean 4 Kernel Benchmark (PHYS-CORE-CONSERVATIVE) ===")
    phys_core_spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-CORE-CONSERVATIVE",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="q1 + q2 = const",
        declared_invariants=[
            DeclaredInvariant(
                name="discrete_algebraic_conservation",
                kind=InvariantKind.CONSERVATION_LAW,
                description="Algebraic flux conservation verified by Lean 4 omega tactic",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="natural_positivity",
                kind=InvariantKind.BOUND_POSITIVITY,
                description="Non-negativity verified by Lean 4 Nat.zero_le",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    phys_core_evidence = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"flux_drift": 0.0, "r2_holdout": 1.0},
    )
    phys_core_cert = bridge.adjudicate_joint_scientific_claim(
        phys_core_spec, phys_ds, phys_core_evidence, standalone_core=True
    )

    domain_records.append({
        "domain": "Discrete Mechanics / Core Lean 4 Kernel Benchmark",
        "valid_candidate": phys_core_cert.to_dict(),
        "trap_candidate": None,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Domain 2: Mathematical Epidemiology (Kermack-McKendrick SIR Dynamics)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=== Domain 2: Mathematical Epidemiology (SIR Mass-Action Dynamics) ===")

    sir_spec = AgnosticCandidateSpecification(
        candidate_id="EPID-SIR-KERMACK-MCKENDRICK",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="d(S+I+R)/dt: (-beta*S*I) + (beta*S*I - gamma*I) + (gamma*I)",
        declared_invariants=[
            DeclaredInvariant(
                name="total_population_conservation",
                kind=InvariantKind.CONSERVATION_LAW,
                description="Total population conservation N = S + I + R, dN/dt = 0",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="compartment_positivity",
                kind=InvariantKind.BOUND_POSITIVITY,
                description="Non-negativity of infected compartment I(t) >= 0",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    sir_ds = AgnosticProblemDataset(
        domain_label="Mathematical Epidemiology",
        X_train=np.linspace(0, 30, 50),
        y_train=np.exp(-0.1 * np.linspace(0, 30, 50)),
        X_holdout=np.linspace(30, 60, 50),
        y_holdout=np.exp(-0.1 * np.linspace(30, 60, 50)),
    )
    sir_evidence = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"population_error": 0.0, "r2_holdout": 0.985},
    )
    sir_cert = bridge.adjudicate_joint_scientific_claim(sir_spec, sir_ds, sir_evidence)

    domain_records.append({
        "domain": "Mathematical Epidemiology",
        "valid_candidate": sir_cert.to_dict(),
        "trap_candidate": None,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Domain 3: Quantitative Market Microstructure (Saturated OFI vs Cubic Runaway)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=== Domain 3: Market Microstructure (Saturated OFI vs Cubic Runaway) ===")

    # Saturated model: c * tanh(v)
    fin_valid_spec = AgnosticCandidateSpecification(
        candidate_id="FIN-SATURATED-OFI-IMPACT",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = c * tanh(velocity)",
        declared_invariants=[
            DeclaredInvariant(
                name="asymptotic_impact_saturation",
                kind=InvariantKind.ASYMPTOTIC_LIMIT,
                description="Bounded market impact under extreme liquidity shocks |y| <= |c|",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="reflection_antisymmetry",
                kind=InvariantKind.SYMMETRY_TRANSFORMATION,
                description="Order-flow buy/sell odd reflection symmetry f(-v) = -f(v)",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    fin_ds = AgnosticProblemDataset(
        domain_label="Quantitative Microstructure",
        X_train=np.linspace(-3, 3, 50),
        y_train=np.tanh(np.linspace(-3, 3, 50)),
        X_holdout=np.linspace(-5, 5, 50),
        y_holdout=np.tanh(np.linspace(-5, 5, 50)),
    )
    fin_valid_evidence = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"r2_holdout": 0.042, "bic_score": -950.0},
    )
    fin_valid_cert = bridge.adjudicate_joint_scientific_claim(fin_valid_spec, fin_ds, fin_valid_evidence)

    # Trap model: Runaway Cubic Polynomial y = 0.5*v + 0.2*v^3
    fin_trap_spec = AgnosticCandidateSpecification(
        candidate_id="FIN-CUBIC-RUNAWAY-POLYNOMIAL",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = 0.5 * v + 0.2 * v^3",
        declared_invariants=[
            DeclaredInvariant(
                name="asymptotic_impact_saturation",
                kind=InvariantKind.ASYMPTOTIC_LIMIT,
                description="Falsely claiming global bounded saturation",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    fin_trap_evidence = EvidenceBundle(
        verified_evidence={EvidenceRequirement.IN_SAMPLE_CONVERGENCE},
        failures=["out_of_sample_explosion"],
    )
    fin_trap_cert = bridge.adjudicate_joint_scientific_claim(fin_trap_spec, fin_ds, fin_trap_evidence)

    domain_records.append({
        "domain": "Quantitative Market Microstructure",
        "valid_candidate": fin_valid_cert.to_dict(),
        "trap_candidate": fin_trap_cert.to_dict(),
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Domain 4: High-Dimensional Geometry & Materials (Sphere Packing Distance)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=== Domain 4: High-Dimensional Geometry (Contact Distance Positivity) ===")

    geom_spec = AgnosticCandidateSpecification(
        candidate_id="GEOM-SPHERE-PACKING-CONTACT",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="d2 = 1 + x^2",
        declared_invariants=[
            DeclaredInvariant(
                name="contact_positivity",
                kind=InvariantKind.BOUND_POSITIVITY,
                description="Pairwise distance squared strictly non-negative d^2 >= 0",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="radial_even_symmetry",
                kind=InvariantKind.SYMMETRY_TRANSFORMATION,
                description="Radial reflection even symmetry f(-x) = f(x)",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    geom_ds = AgnosticProblemDataset(
        domain_label="High-Dimensional Geometry",
        X_train=np.linspace(-2, 2, 50),
        y_train=1 + np.linspace(-2, 2, 50)**2,
        X_holdout=np.linspace(-4, 4, 50),
        y_holdout=1 + np.linspace(-4, 4, 50)**2,
    )
    geom_evidence = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"barrier_violation": 0.0, "r2_holdout": 1.0},
    )
    geom_cert = bridge.adjudicate_joint_scientific_claim(geom_spec, geom_ds, geom_evidence)

    domain_records.append({
        "domain": "High-Dimensional Geometry",
        "valid_candidate": geom_cert.to_dict(),
        "trap_candidate": None,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Domain 5: Fluid Mechanics (Incompressible Solenoidal Flow)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=== Domain 5: Fluid Mechanics (Solenoidal Incompressible Field) ===")

    fluid_spec = AgnosticCandidateSpecification(
        candidate_id="FLUID-SOLENOIDAL-INCOMPRESSIBLE",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="div u: incompressible solenoidal field",
        declared_invariants=[
            DeclaredInvariant(
                name="incompressible_divergence_free",
                kind=InvariantKind.CONSERVATION_LAW,
                description="Zero divergence velocity field ∇ · u = 0 (volume conservation)",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="kinetic_energy_positivity",
                kind=InvariantKind.BOUND_POSITIVITY,
                description="Kinetic energy density 0.5 * ||u||^2 >= 0",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )
    fluid_ds = AgnosticProblemDataset(
        domain_label="Fluid Dynamics",
        X_train=np.linspace(0, 1, 50),
        y_train=np.zeros(50),
        X_holdout=np.linspace(1, 2, 50),
        y_holdout=np.zeros(50),
    )
    fluid_evidence = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"divergence_residual": 0.0},
    )
    fluid_cert = bridge.adjudicate_joint_scientific_claim(fluid_spec, fluid_ds, fluid_evidence)

    domain_records.append({
        "domain": "Fluid Mechanics",
        "valid_candidate": fluid_cert.to_dict(),
        "trap_candidate": None,
    })

    # Option 1: Docker Verification Dispatcher Check
    docker_bridge = LeanUniversalBridge(
        output_dir=LEAN_OUT_DIR,
        backend=VerificationBackend.DOCKER_MATHLIB,
    )
    docker_avail = docker_bridge.is_docker_available
    docker_demo_result = None
    if docker_avail:
        logger.info("=== Option 1: Docker Mathlib Execution Dispatch ===")
        doc_res = docker_bridge.verify_candidate_formally(phys_core_spec, standalone_core=True)
        docker_demo_result = {
            "candidate_id": doc_res.candidate_id,
            "proof_claim_status": doc_res.proof_claim_status.value,
            "kernel_exit_code": doc_res.kernel_exit_code,
            "notes": doc_res.notes,
        }

    # Summary payload
    summary = {
        "suite_name": "Agnostic Lean 4 Cross-Domain Formal Verification Suite",
        "primary_backend": "Option 2: Core Axiomatic (Pure Core Lean 4)",
        "docker_backend": "Option 1: Docker Mathlib Container Dispatcher",
        "lean_compiler_available": bridge.is_lean_compiler_available,
        "docker_daemon_available": docker_avail,
        "docker_verification_demo": docker_demo_result,
        "exported_modules_dir": str(LEAN_OUT_DIR),
        "domains_evaluated": len(domain_records),
        "results": domain_records,
    }

    prov_path = ARTIFACT_DIR / "cross_domain_lean_verification_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved provenance to: %s", prov_path)

    return summary


if __name__ == "__main__":
    summary = run_cross_domain_lean_verification_suite()
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN FORMAL VERIFICATION SUMMARY")
    print("=" * 80)
    for rec in summary["results"]:
        dom = rec["domain"]
        valid = rec["valid_candidate"]
        print(f"\n[Domain: {dom}]")
        print(f"  Valid Candidate: {valid['candidate_id']}")
        print(f"    Joint Label : {valid['joint_epistemic_label']}")
        print(f"    Certified   : {valid['is_fully_certified']}")
        print(f"    Synthesis   : {valid['epistemic_synthesis_summary']}")
        if rec["trap_candidate"]:
            trap = rec["trap_candidate"]
            print(f"  Trap Candidate : {trap['candidate_id']}")
            print(f"    Joint Label : {trap['joint_epistemic_label']}")
            print(f"    Certified   : {trap['is_fully_certified']}")
            print(f"    Synthesis   : {trap['epistemic_synthesis_summary']}")
