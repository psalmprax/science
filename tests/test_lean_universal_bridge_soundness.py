"""Soundness regression tests for LeanUniversalBridge."""
from __future__ import annotations

from apedp_dyn.epistemic.agnostic_adapter import (
    AgnosticCandidateSpecification,
    DeclaredInvariant,
    InvariantKind,
)
from apedp_dyn.epistemic.claim_ladder import ClaimLevel
from apedp_dyn.epistemic.lean_universal_bridge import (
    LeanUniversalBridge,
    VerificationBackend,
    _contains_sorry_or_admit,
)
from apedp_dyn.epistemic.proof_claim import ProofClaimStatus


def test_rational_bounded_denominator_synthesizes_without_sorry() -> None:
    bridge = LeanUniversalBridge(backend=VerificationBackend.LOCAL_MATHLIB)
    cand = AgnosticCandidateSpecification(
        candidate_id="rational_bound_ok",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = v / (1 + Abs(s))",
        declared_invariants=[
            DeclaredInvariant(
                name="rational_bound",
                kind=InvariantKind.ASYMPTOTIC_LIMIT,
                description="Bounded rational damping",
            )
        ],
    )
    code, thms = bridge.generate_lean4_module(cand, standalone_core=False)
    assert len(thms) == 1
    assert thms[0].is_formally_proven is True
    assert not _contains_sorry_or_admit(code)
    assert "sorry" not in thms[0].lean_proof_script.lower()
    assert "abs_div" in thms[0].lean_proof_script


def test_offline_sorry_admit_and_sorryax_always_rejected() -> None:
    bridge = LeanUniversalBridge(backend=VerificationBackend.LOCAL_MATHLIB)
    bridge._lean_binary = None

    for snippet in (
        "theorem t1 : 1 = 1 := by\n  sorry\n",
        "theorem t2 : 1 = 1 := by admit\n",
        "theorem t3 : 1 = 1 := sorryAx _\n",
    ):
        res = bridge.verify_raw_lean_code_in_memory(snippet, candidate_id="incomplete")
        assert res.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
        assert res.all_theorems_proven is False

    # Candidate with unproved invariant must also be rejected offline when sorry is emitted.
    bad_cand = AgnosticCandidateSpecification(
        candidate_id="deg6_offline_reject",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = x^6 + 1",
        declared_invariants=[
            DeclaredInvariant(
                name="bound_check",
                kind=InvariantKind.ASYMPTOTIC_LIMIT,
            )
        ],
    )
    res_cand = bridge.verify_candidate_formally(bad_cand, standalone_core=False, in_memory=True)
    assert res_cand.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
    assert res_cand.all_theorems_proven is False

    # Comments mentioning sorry must not false-positive when proof itself is clean.
    assert not _contains_sorry_or_admit("/- no sorry used -/\ntheorem ok : 1 = 1 := by\n  -- sorry avoided\n  rfl\n")


def test_eliminates_unbound_symbols_and_vacuous_tautologies() -> None:
    bridge = LeanUniversalBridge(backend=VerificationBackend.LOCAL_MATHLIB)

    # Polynomial x + 1 must NOT trigger trig Pythagorean theorem or positivity.
    poly_cand = AgnosticCandidateSpecification(
        candidate_id="poly_no_cheat",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = x + 1",
        declared_invariants=[
            DeclaredInvariant(name="cons", kind=InvariantKind.CONSERVATION_LAW),
            DeclaredInvariant(name="pos", kind=InvariantKind.BOUND_POSITIVITY),
            DeclaredInvariant(name="bound", kind=InvariantKind.ASYMPTOTIC_LIMIT),
        ],
    )
    code, thms = bridge.generate_lean4_module(poly_cand, standalone_core=False)
    assert "Energy" not in code
    assert "sin_sq_add_cos_sq" not in code
    assert all(not t.is_formally_proven for t in thms)

    # Structural sum of squares is proved wrt the actual candidate expression.
    sos_cand = AgnosticCandidateSpecification(
        candidate_id="sos_pos",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = x^2 + 3 * z^2 + 1",
        declared_invariants=[
            DeclaredInvariant(name="energy_positivity", kind=InvariantKind.BOUND_POSITIVITY)
        ],
    )
    code_sos, thms_sos = bridge.generate_lean4_module(sos_cand, standalone_core=False)
    assert thms_sos[0].is_formally_proven is True
    assert "positivity" in thms_sos[0].lean_proof_script
    assert "0 : ℝ) ≤ (0 : ℝ)" not in code_sos
    assert "x" in thms_sos[0].lean_theorem_statement and "z" in thms_sos[0].lean_theorem_statement

    # Odd parity symmetry binds actual variables instead of unbound `f`.
    odd_cand = AgnosticCandidateSpecification(
        candidate_id="odd_cubic",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = x^3 + 2*x",
        declared_invariants=[
            DeclaredInvariant(name="symm", kind=InvariantKind.SYMMETRY_TRANSFORMATION)
        ],
    )
    code_odd, thms_odd = bridge.generate_lean4_module(odd_cand, standalone_core=False)
    assert thms_odd[0].is_formally_proven is True
    assert "f (-x)" not in code_odd
    assert "(x) ^ 3" in thms_odd[0].lean_theorem_statement


def test_infer_invariants_ast_first() -> None:
    bridge = LeanUniversalBridge(backend=VerificationBackend.LOCAL_MATHLIB)
    tanh_invs = bridge.infer_invariants("c1", "alpha * tanh(beta * z)")
    assert any(i.kind == InvariantKind.ASYMPTOTIC_LIMIT for i in tanh_invs)
    assert any(i.kind == InvariantKind.SYMMETRY_TRANSFORMATION for i in tanh_invs)

    rat_invs = bridge.infer_invariants("c2", "u / (2 + Abs(w))")
    assert any(i.kind == InvariantKind.ASYMPTOTIC_LIMIT for i in rat_invs)

    deg6_invs = bridge.infer_invariants("c3", "x^6 + 2*x")
    assert any(i.kind == InvariantKind.ASYMPTOTIC_LIMIT for i in deg6_invs)
    res = bridge.verify_equation_in_memory(
        "x^6 + 2*x",
        candidate_id="c3",
        declared_invariants=deg6_invs,
        standalone_core=False,
    )
    assert res.proof_claim_status == ProofClaimStatus.P0_NO_PROOF
