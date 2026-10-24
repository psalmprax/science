"""Tests for Agnostic / Universal Lean 4 Formal Verification Bridge.

Verifies:
  - Strict Epistemic Proof Tiers:
      P0: Refuted by counterexample / dissipation / runaway, or compiler failed with proof error
      P1: Deductive proof candidate synthesized in Lean 4 code, awaiting compiler kernel check
      P2: Formally verified proof (Lean 4 compiler kernel executed and exited with code 0)
  - Mathlib-targeted modules unlinked in standalone mode report P1_CANDIDATE, NEVER P2_VERIFIED
  - Core Lean 4 modules genuinely executed by `lean` binary achieve true P2_VERIFIED
  - Detection of mathematical flaws (Energy leak traps, polynomial runaway) -> P0_NO_PROOF
  - Joint Epistemic Adjudication: (Empirical Claim Level, Formal Proof Status)
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from apedp.mathematical.mathematical_layer import ProofClaimStatus
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
)


@pytest.fixture
def bridge(tmp_path):
    return LeanUniversalBridge(output_dir=tmp_path / "lean_test_out")


def test_mathlib_candidate_uncompiled_reports_p1_candidate(bridge):
    """Case 1: Mathlib-targeted module in standalone invocation reports P1_CANDIDATE (never overclaims to P2)."""
    inv_energy = DeclaredInvariant(
        name="hamiltonian_energy_conservation",
        kind=InvariantKind.CONSERVATION_LAW,
        description="Preservation of mechanical energy H = 0.5*(v^2 + x^2)",
        required_for_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
    )

    spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-OSCILLATOR-MATHLIB",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        formula_repr="x(t) = sin(t)",
        declared_invariants=[inv_energy],
    )

    res = bridge.verify_candidate_formally(spec)

    assert res.candidate_id == "PHYS-OSCILLATOR-MATHLIB"
    # When Lean runs on a file importing Mathlib without lake mathlib build, compiler exits with code 1:
    assert res.proof_claim_status == ProofClaimStatus.P1_CANDIDATE
    assert res.all_theorems_proven is False
    assert res.kernel_exit_code == 1
    assert "Mathlib library environment required" in res.notes
    assert "import Mathlib.Data.Real.Basic" in res.lean_source_code


def test_core_lean4_kernel_execution_achieves_p2_verified(bridge):
    """Case 2: Pure Core Lean 4 module executed by compiler kernel achieves genuine P2_VERIFIED."""
    inv_energy = DeclaredInvariant(
        name="discrete_algebraic_conservation",
        kind=InvariantKind.CONSERVATION_LAW,
        description="Algebraic flux conservation verified by Lean 4 omega tactic",
        required_for_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
    )
    inv_pos = DeclaredInvariant(
        name="natural_positivity",
        kind=InvariantKind.BOUND_POSITIVITY,
        description="Non-negativity verified by Lean 4 Nat.zero_le",
        required_for_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
    )

    spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-CORE-CONSERVATIVE",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        formula_repr="q1 + q2 = const",
        declared_invariants=[inv_energy, inv_pos],
    )

    # Execute standalone Core Lean 4 verification
    res = bridge.verify_candidate_formally(spec, standalone_core=True)

    if bridge.is_lean_compiler_available:
        # Compiler genuinely ran and accepted the proof!
        assert res.kernel_exit_code == 0
        assert res.proof_claim_status == ProofClaimStatus.P2_VERIFIED
        assert res.all_theorems_proven is True
        assert "exit code 0" in res.notes
    else:
        # If compiler binary was absent, must be P1, NEVER P2
        assert res.proof_claim_status == ProofClaimStatus.P1_CANDIDATE


def test_physics_energy_leak_trap_refuted(bridge):
    """Case 3: Energy-leaking oscillator (dissipation factor) is caught and refuted -> P0_NO_PROOF."""
    inv_energy = DeclaredInvariant(
        name="hamiltonian_energy_conservation",
        kind=InvariantKind.CONSERVATION_LAW,
        description="Preservation of mechanical energy",
        required_for_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
    )

    spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-ENERGY-LEAK-TRAP",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        formula_repr="x(t) = sin(t) * exp(-0.02*t)",
        declared_invariants=[inv_energy],
    )

    res = bridge.verify_candidate_formally(spec)

    assert res.all_theorems_proven is False
    assert res.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert res.theorems[0].is_formally_proven is False
    assert "dissipation" in res.theorems[0].diagnostic_message.lower()


def test_finance_cubic_runaway_refuted(bridge):
    """Case 4: Cubic polynomial runaway claiming global asymptotic bound is formally refuted -> P0_NO_PROOF."""
    inv_bound = DeclaredInvariant(
        name="asymptotic_saturation",
        kind=InvariantKind.ASYMPTOTIC_LIMIT,
        description="Claiming bounded response under extreme market shock",
        required_for_tier=ClaimLevel.C3_STRUCTURAL,
    )
    spec = AgnosticCandidateSpecification(
        candidate_id="FIN-RUNAWAY-CUBIC",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = 0.5 * v + 0.2 * v^3",
        declared_invariants=[inv_bound],
    )
    res = bridge.verify_candidate_formally(spec)
    assert res.all_theorems_proven is False
    assert res.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert "polynomial_runaway" in res.theorems[0].proof_method
    assert "FALSIFIED" in res.theorems[0].lean_proof_script


def test_joint_epistemic_adjudication_candidate_vs_verified(bridge):
    """Case 5: Joint referee distinguishes between P1_CANDIDATE (unverified by kernel) and P2_VERIFIED."""
    inv_cons = DeclaredInvariant(
        name="conservation",
        kind=InvariantKind.CONSERVATION_LAW,
        required_for_tier=ClaimLevel.C3_STRUCTURAL,
    )

    spec = AgnosticCandidateSpecification(
        candidate_id="MODEL-EPI-TEST",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="x(t) = sin(t)",
        declared_invariants=[inv_cons],
    )

    ds = AgnosticProblemDataset(
        domain_label="Physics",
        X_train=np.linspace(0, 10, 50),
        y_train=np.sin(np.linspace(0, 10, 50)),
        X_holdout=np.linspace(10, 20, 50),
        y_holdout=np.sin(np.linspace(10, 20, 50)),
    )

    # Full empirical evidence satisfying C3
    bundle = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
            EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
            EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED,
        },
        metrics={"bic_score": -1200.0, "tau_epi": 1e-3},
    )

    # 1. Uncompiled Mathlib module -> P1_CANDIDATE
    cert_p1 = bridge.adjudicate_joint_scientific_claim(spec, ds, bundle, standalone_core=False)
    assert cert_p1.empirical_claim_level == ClaimLevel.C3_STRUCTURAL
    assert cert_p1.formal_proof_status == ProofClaimStatus.P1_CANDIDATE
    assert cert_p1.is_fully_certified is False  # Cannot fully certify without kernel check!
    assert cert_p1.joint_epistemic_label == "(C3_STRUCTURAL, P1_CANDIDATE)"
    assert "pending compiler kernel verification" in cert_p1.epistemic_synthesis_summary

    # 2. Kernel-compiled Core Lean 4 module -> P2_VERIFIED
    if bridge.is_lean_compiler_available:
        cert_p2 = bridge.adjudicate_joint_scientific_claim(spec, ds, bundle, standalone_core=True)
        assert cert_p2.empirical_claim_level == ClaimLevel.C3_STRUCTURAL
        assert cert_p2.formal_proof_status == ProofClaimStatus.P2_VERIFIED
        assert cert_p2.is_fully_certified is True
        assert cert_p2.joint_epistemic_label == "(C3_STRUCTURAL, P2_VERIFIED)"
        assert "FULLY CERTIFIED SCIENTIFIC DISCOVERY" in cert_p2.epistemic_synthesis_summary


def test_joint_epistemic_adjudication_empirical_failure(bridge):
    """Case 6: A model with valid math but failing holdout evidence is rejected empirically."""
    inv_pos = DeclaredInvariant(
        name="positivity",
        kind=InvariantKind.BOUND_POSITIVITY,
        required_for_tier=ClaimLevel.C3_STRUCTURAL,
    )

    spec = AgnosticCandidateSpecification(
        candidate_id="MODEL-VALID-MATH-BAD-FIT",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = 1 + x^2",
        declared_invariants=[inv_pos],
    )

    ds = AgnosticProblemDataset(
        domain_label="Test",
        X_train=np.linspace(0, 10, 50),
        y_train=np.zeros(50),
        X_holdout=np.linspace(10, 20, 50),
        y_holdout=np.zeros(50),
    )

    # Failing empirical holdout evidence (missing out of sample holdout and rivals)
    bundle = EvidenceBundle(
        verified_evidence={
            EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        },
        failures=["holdout_divergence"],
    )

    cert = bridge.adjudicate_joint_scientific_claim(spec, ds, bundle, standalone_core=True)

    assert cert.is_fully_certified is False
    assert cert.empirical_claim_level == ClaimLevel.C1_DESCRIPTIVE  # Downgraded to C1
    assert "DOWNGRADED/REFUSED" in cert.epistemic_synthesis_summary


def test_option2_axiomatic_core_continuous_models_achieve_p2(bridge):
    """Option 2: Continuous models verified via lightweight axiomatic signatures in Core Lean 4 achieve P2."""
    fin_spec = AgnosticCandidateSpecification(
        candidate_id="FIN-SATURATED-OFI-IMPACT",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = c * tanh(velocity)",
        declared_invariants=[
            DeclaredInvariant(
                name="asymptotic_impact_saturation",
                kind=InvariantKind.ASYMPTOTIC_LIMIT,
                description="Bounded market impact |y| <= |c|",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
            DeclaredInvariant(
                name="reflection_antisymmetry",
                kind=InvariantKind.SYMMETRY_TRANSFORMATION,
                description="Odd reflection symmetry f(-v) = -f(v)",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )

    res = bridge.verify_candidate_formally(fin_spec, standalone_core=True)
    if bridge.is_lean_compiler_available:
        assert res.kernel_exit_code == 0
        assert res.proof_claim_status == ProofClaimStatus.P2_VERIFIED
        assert res.all_theorems_proven is True
        assert "Option 2: Pure Core Lean 4 Axiomatic Infrastructure" in res.lean_source_code


def test_option1_docker_mathlib_execution(tmp_path):
    """Option 1: Docker containerized Lean dispatcher execution verification."""
    from apedp_dyn.epistemic.lean_universal_bridge import VerificationBackend

    scratch_dir = tmp_path / "lean_docker_out"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    bridge_docker = LeanUniversalBridge(
        output_dir=scratch_dir,
        backend=VerificationBackend.DOCKER_MATHLIB,
    )

    if not bridge_docker.is_docker_available:
        pytest.skip("Docker daemon not available in current test environment")
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        pytest.skip("Docker CLI not available")
    inspect = subprocess.run(
        [docker_bin, "image", "inspect", "leanprovercommunity/mathlib:latest"],
        capture_output=True,
        timeout=15,
        check=False,
    )
    if inspect.returncode != 0:
        pytest.skip(
            "leanprovercommunity/mathlib:latest is not pulled locally; "
            "the bridge's 25s execution budget cannot include a multi-GB image pull. "
            "Run `docker pull leanprovercommunity/mathlib:latest` to enable this test."
        )

    spec = AgnosticCandidateSpecification(
        candidate_id="PHYS-DOCKER-CONSERVATIVE",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="q1 + q2 = const",
        declared_invariants=[
            DeclaredInvariant(
                name="discrete_algebraic_conservation",
                kind=InvariantKind.CONSERVATION_LAW,
                description="Algebraic flux conservation verified inside Docker",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            ),
        ],
    )

    res = bridge_docker.verify_candidate_formally(spec, standalone_core=True)
    assert res.kernel_exit_code == 0
    assert res.proof_claim_status == ProofClaimStatus.P2_VERIFIED
    assert res.all_theorems_proven is True
    assert "inside Docker container" in res.notes


def test_in_memory_equation_verification_zero_disk_files():
    """Verify arbitrary equations directly in memory without writing any files to disk."""
    from apedp_dyn.epistemic.lean_universal_bridge import verify_equation_in_memory

    # Test 1: Saturated market impact equation
    res_tanh = verify_equation_in_memory("y = c * tanh(velocity)", candidate_id="SATURATED_FLOW")
    assert res_tanh.proof_claim_status == ProofClaimStatus.P2_VERIFIED
    assert res_tanh.all_theorems_proven is True
    assert res_tanh.kernel_exit_code == 0
    assert res_tanh.lean_file_path is None
    assert "in-memory stdin streaming" in res_tanh.notes
    assert len(res_tanh.theorems) == 2

    # Test 2: Linear superposition flux balance
    res_lin = verify_equation_in_memory("dx_dt = -gamma * x", candidate_id="LINEAR_SUPERPOSITION_BALANCE")
    assert res_lin.proof_claim_status == ProofClaimStatus.P2_VERIFIED
    assert res_lin.all_theorems_proven is True
    assert res_lin.kernel_exit_code == 0
    assert res_lin.lean_file_path is None


def test_in_memory_equation_trap_refuted():
    """Verify in-memory verification refutes mathematical traps without touching disk."""
    from apedp_dyn.epistemic.lean_universal_bridge import verify_equation_in_memory

    # Trap 1: Cubic runaway polynomial
    res_cubic = verify_equation_in_memory("y = 0.5 * v + 0.2 * v^3 runaway", candidate_id="FIN-RUNAWAY")
    assert res_cubic.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert res_cubic.all_theorems_proven is False
    assert res_cubic.lean_file_path is None

    # Trap 2: Dissipative leak claiming conservation
    res_leak = verify_equation_in_memory("x(t) = sin(t) * exp(-0.02*t) leak", candidate_id="PHYS-LEAK")
    assert res_leak.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert res_leak.all_theorems_proven is False
    assert res_leak.lean_file_path is None

    # Trap 3: Dissipative flow claiming conservation, at a decay rate OUTSIDE
    # every legacy string trigger (0.05 is not "exp(-0.1"/"exp(-0.2"/"exp(-1").
    # Refutation is AST-derived: the exp(-c*t) damping factor is read from the
    # parsed formula, so ANY positive rate is caught. (The previous fixture
    # relied on the substring "dissipat" in the candidate id, which was the
    # test-case rigging the remediation removed; the identical formula
    # "dx_dt = -gamma * x" without the trap name follows the ordinary
    # linear-flux-balance path, as `LINEAR_SUPERPOSITION_BALANCE` does above.)
    res_dissip = verify_equation_in_memory("x(t) = exp(-0.05 * t)", candidate_id="DISSIPATIVE_FLOW_TRAP")
    assert res_dissip.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert res_dissip.all_theorems_proven is False
    assert res_dissip.lean_file_path is None


def test_in_memory_raw_lean_code_verification():
    """Verify raw Lean 4 code strings directly via stdin without touching disk."""
    from apedp_dyn.epistemic.lean_universal_bridge import verify_raw_lean_code_in_memory

    # Valid proof term
    valid_code = "theorem tautology (p : Prop) (h : p) : p := h\n"
    res_valid = verify_raw_lean_code_in_memory(valid_code, candidate_id="tautology")
    assert res_valid.proof_claim_status == ProofClaimStatus.P2_VERIFIED
    assert res_valid.all_theorems_proven is True
    assert res_valid.kernel_exit_code == 0
    assert res_valid.lean_file_path is None

    # Invalid / sorry trap
    sorry_code = "theorem false_claim : 1 = 2 := by sorry\n"
    res_sorry = verify_raw_lean_code_in_memory(sorry_code, candidate_id="sorry_claim")
    assert res_sorry.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert res_sorry.all_theorems_proven is False
    assert res_sorry.lean_file_path is None

