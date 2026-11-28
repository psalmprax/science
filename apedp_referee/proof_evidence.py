"""What a formal-verification run reports, as plain data.

Stage 1 of the promotion lifecycle judges a :class:`LeanVerificationResult`:
the kernel's exit code, whether the module declared its own axioms, whether a
dimensional and a bound theorem were proven. These types used to be defined
where results are *produced* -- ``InvariantKind`` in ``agnostic_adapter`` and
the two result records in ``lean_universal_bridge``, which generates Lean,
runs the toolchain, and imports SymPy and (through ``apedp.safety``) the whole
legacy ``apedp`` package. The referee could not name the evidence it consumes
without importing the machinery that makes it, and that machinery's
dependencies with it.

They live here now, with the standard library as their only dependency, for
the same reason ``proof_claim.py`` exists. ``agnostic_adapter`` and
``lean_universal_bridge`` import them from here and re-export them, so every
existing ``from ...agnostic_adapter import InvariantKind`` and
``from ...lean_universal_bridge import LeanVerificationResult`` resolves to
the same class objects as before.

Nothing in this module verifies anything. A result object is a claim about a
verification run; ``PromotionTracker.verify_stage_1_from_lean`` decides what
the claim is worth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from apedp_referee.proof_claim import ProofClaimStatus

__all__ = ["InvariantKind", "LeanTheoremVerification", "LeanVerificationResult"]


class InvariantKind(str, Enum):
    """Generic taxonomy of scientific invariants that any domain can declare."""
    CONSERVATION_LAW = "conservation_law"         # Delta Q <= epsilon (mass, energy, momentum, flux)
    SYMMETRY_TRANSFORMATION = "symmetry_transform" # f(T(x)) == rho(T) f(x) (parity, rotation, translation)
    BOUND_POSITIVITY = "bound_positivity"         # y >= y_min, L <= y <= U (non-negativity, probability)
    MONOTONICITY = "monotonicity"                 # dy/dx >= 0 or order-preserving (entropy, priority)
    LIPSCHITZ_SMOOTHNESS = "lipschitz_smoothness" # ||f(x1) - f(x2)|| <= L ||x1 - x2||
    SCALE_HOMOGENEITY = "scale_homogeneity"       # f(lambda * x) == lambda^k * f(x)
    ASYMPTOTIC_LIMIT = "asymptotic_limit"         # lim_{x -> inf} f(x) == C (saturation, bounded tail)
    INTERVENTIONAL_INVARIANCE = "interventional_inv" # Invariance under targeted do(X) perturbation
    CUSTOM_PREDICATE = "custom_predicate"         # Arbitrary user/problem-supplied verification callable


@dataclass(frozen=True)
class LeanTheoremVerification:
    invariant_name: str
    invariant_kind: InvariantKind
    theorem_identifier: str
    lean_theorem_statement: str
    lean_proof_script: str
    is_formally_proven: bool
    proof_method: str  # e.g., "lean4_kernel", "symbolic_deduction", "counterexample_refuted"
    diagnostic_message: str
    # True when the proof is discharged against an axiom this module declared.
    # Such a record is a schema check, never an independent derivation, and the
    # verifier caps the candidate at P1 whenever one is present.
    axiom_dependent: bool = False


_UNPROVEN = re.compile(r"(?<![\w.'])(?:sorry|admit|sorryAx)(?![\w'])")
_TRUSTS_COMPILED_CODE = re.compile(r"(?<![\w'])(?:native_decide|ofReduceBool|reduceBool)(?![\w'])")   # also as Lean.ofReduceBool


def lean_code_only(source: str) -> str:
    """A Lean source with its comments and text literals removed.

    ``--`` to the end of a line, ``/- ... -/`` (nested, doc comments included) and ``"..."`` are dropped, so that a
    word inside a comment or a string is not read as code, and code is not hidden from the reading by one.
    """
    out: list[str] = []
    i, n, depth = 0, len(source), 0
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if depth:
            if ch == "/" and nxt == "-":
                depth, i = depth + 1, i + 2
            elif ch == "-" and nxt == "/":
                depth, i = depth - 1, i + 2
            else:
                if ch == "\n":
                    out.append("\n")
                i += 1
        elif ch == "/" and nxt == "-":
            depth, i = 1, i + 2
        elif ch == "-" and nxt == "-":
            while i < n and source[i] != "\n":
                i += 1
        elif ch == '"':
            i += 1
            while i < n and source[i] != '"':
                i += 2 if source[i] == "\\" else 1
            i += 1
            out.append(" ")
        elif ch == "'" and re.match(r"'(?:\\.|[^\\'])'", source[i:i + 4]):
            i += len(re.match(r"'(?:\\.|[^\\'])'", source[i:i + 4]).group(0))  # type: ignore[union-attr]
            out.append(" ")                                  # a character literal such as '"' is not the start of a string
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def lean_source_objections(source: str) -> list[str]:
    """Why this Lean source cannot stand behind a verified (P2) claim; empty if nothing in its text says so.

    Read outside comments and text literals. An empty source is an objection: there is nothing to have checked.
    ``sorry``, ``admit`` and ``sorryAx`` leave a goal unproven. ``native_decide`` and the axioms under it
    (``ofReduceBool``, ``reduceBool``) trust compiled code in place of the kernel. Until 28 November 2026 Stage 1
    looked for the word ``sorry`` anywhere in the text and for nothing else, and accepted an empty source.

    A declared ``axiom`` is not an objection here: the bridge's modules declare the real field that way, and a
    theorem that rests on one is marked ``axiom_dependent``, which Stage 1 already refuses theorem by theorem.
    """
    code = lean_code_only(source or "")
    if not code.strip():
        return ["no Lean source outside comments: nothing was there to check"]
    out = []
    if _UNPROVEN.search(code):
        out.append("a goal is left unproven (sorry, admit or sorryAx)")
    if _TRUSTS_COMPILED_CODE.search(code):
        out.append("a proof trusts compiled code in place of the kernel (native_decide)")
    return out


@dataclass
class LeanVerificationResult:
    candidate_id: str
    formula_repr: str
    all_theorems_proven: bool
    proof_claim_status: ProofClaimStatus  # P0, P1, P2
    theorems: list[LeanTheoremVerification] = field(default_factory=list)
    lean_source_code: str = ""
    lean_file_path: Path | None = None
    kernel_exit_code: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "formula_repr": self.formula_repr,
            "all_theorems_proven": self.all_theorems_proven,
            "proof_claim_status": self.proof_claim_status.value,
            "theorems": [
                {
                    "invariant_name": t.invariant_name,
                    "invariant_kind": t.invariant_kind.value,
                    "theorem_identifier": t.theorem_identifier,
                    "lean_statement": t.lean_theorem_statement,
                    "proof_method": t.proof_method,
                    "is_formally_proven": t.is_formally_proven,
                    "axiom_dependent": t.axiom_dependent,
                    "diagnostic": t.diagnostic_message,
                }
                for t in self.theorems
            ],
            "lean_file_path": str(self.lean_file_path) if self.lean_file_path else None,
            "notes": self.notes,
        }
