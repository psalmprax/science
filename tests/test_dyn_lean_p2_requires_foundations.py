"""P2 requires a derivation from Lean's foundations, not from the file's own axioms.

The core-axiomatic mode (Option 2, no Mathlib) emits a module that declares
``axiom abs_mul``, ``axiom tanh_odd``, ``axiom cube_neg`` and friends, then
"proves" the invariant with ``rw [abs_mul ...]``. The kernel accepts it, because
the conclusion is embedded in the premises. Before this pin, that acceptance was
reported as P2_VERIFIED, and 94 theorems in ``lean_formal/`` were counted on it.

The decision is exercised with the kernel call stubbed to exit 0, so the test is
about the status rule rather than about whether Lean is installed on the machine
running it -- the environment in which the overclaim actually occurred.
"""
from __future__ import annotations

import types

import pytest

from apedp.mathematical.mathematical_layer import ProofClaimStatus
from apedp_dyn.epistemic import lean_universal_bridge as bridge_module
from apedp_dyn.epistemic.lean_universal_bridge import (
    AgnosticCandidateSpecification,
    ClaimLevel,
    DeclaredInvariant,
    InvariantKind,
    LeanUniversalBridge,
    _declares_axioms,
)

HONEST_PROOF = "theorem probe : (1 : Nat) = 1 := rfl\n"
AXIOM_DEPENDENT = "axiom cheat : (1 : Nat) = 2\ntheorem probe : (1 : Nat) = 2 := cheat\n"


@pytest.fixture
def kernel_accepts_everything(monkeypatch):
    """A Lean kernel that exits 0 without reading the module.

    This is the harshest case for the rule: everything compiles, so the only
    thing that can keep an axiom-dependent module out of P2 is the status rule
    itself. `_lean_binary` is assigned per instance because it is resolved in
    `__init__` from the interpreter's PATH.
    """
    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bridge_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        LeanUniversalBridge,
        "is_lean_compiler_available",
        property(lambda self: True),
    )

    def make_bridge() -> LeanUniversalBridge:
        bridge = LeanUniversalBridge()
        bridge._lean_binary = "lean"
        return bridge

    return make_bridge


def _candidate() -> AgnosticCandidateSpecification:
    return AgnosticCandidateSpecification(
        candidate_id="p2_cap_probe",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = v * tanh(s)",
        declared_invariants=[
            DeclaredInvariant(
                name="asymptotic_saturation",
                kind=InvariantKind.ASYMPTOTIC_LIMIT,
                description="|c * tanh v| <= |c| for non-negative |c|",
                required_for_tier=ClaimLevel.C3_STRUCTURAL,
            )
        ],
    )


def test_a_module_that_declares_its_own_axioms_is_detected():
    code, theorems = LeanUniversalBridge().generate_lean4_module(
        _candidate(), standalone_core=True)
    assert _declares_axioms(code)
    assert "axiom abs_mul" in code
    assert theorems and theorems[0].axiom_dependent is True
    assert theorems[0].proof_method.startswith("syntactic_schema_check::")


def test_a_mathlib_module_declares_no_axioms_of_its_own():
    code, theorems = LeanUniversalBridge().generate_lean4_module(
        _candidate(), standalone_core=False)
    assert not _declares_axioms(code)
    assert "import Mathlib" in code
    assert theorems and theorems[0].axiom_dependent is False


def test_kernel_acceptance_of_an_axiom_free_module_is_still_p2(kernel_accepts_everything):
    result = kernel_accepts_everything().verify_raw_lean_code_in_memory(HONEST_PROOF)
    assert result.proof_claim_status is ProofClaimStatus.P2_VERIFIED
    assert result.all_theorems_proven is True


def test_kernel_acceptance_of_a_self_axiomatised_module_is_capped_at_p1(kernel_accepts_everything):
    result = kernel_accepts_everything().verify_raw_lean_code_in_memory(AXIOM_DEPENDENT)
    assert result.proof_claim_status is ProofClaimStatus.P1_CANDIDATE
    assert result.all_theorems_proven is False
    assert "P1, not P2" in result.notes
    assert result.kernel_exit_code == 0


def test_the_candidate_path_cannot_reach_p2_on_injected_axioms(kernel_accepts_everything):
    """The path the 94 `lean_formal/` theorems travelled."""
    result = kernel_accepts_everything().verify_candidate_formally(
        candidate=_candidate(), standalone_core=True, in_memory=True)
    assert result.kernel_exit_code == 0
    assert result.lean_source_code and _declares_axioms(result.lean_source_code)
    assert result.proof_claim_status is ProofClaimStatus.P1_CANDIDATE
    assert result.all_theorems_proven is False
    assert "locally declared axioms" in result.notes


def test_the_record_serialises_the_flag():
    _code, theorems = LeanUniversalBridge().generate_lean4_module(
        _candidate(), standalone_core=True)
    payload = theorems[0]
    assert payload.axiom_dependent is True
    result = LeanUniversalBridge().verify_candidate_formally(
        candidate=_candidate(), standalone_core=True, in_memory=True)
    assert result.to_dict()["theorems"][0]["axiom_dependent"] is True
