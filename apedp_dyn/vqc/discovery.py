"""Discovery admission: a discovered circuit cannot be promoted without a receipt.

Status: experimental

:mod:`apedp_dyn.quantum_compilation.search` and the discovery engines in
``apedp/`` do not emit this IR, so nothing they propose can be adjudicated yet. This
module is the interface that changes that: a :class:`CircuitCandidate` names a
candidate circuit, the reference it claims to replace, the resource claims it is
offered under, and optionally the output distribution it says it reproduces.

Adjudication is not a new referee. It routes to the two that exist and combines
their verdicts:

* **equivalence** -- :func:`apedp_dyn.vqc.equivalence.adjudicate_clifford_equivalence`
  when both circuits are Clifford (exact at any size), otherwise
  :func:`apedp_dyn.vqc.equivalence.adjudicate_equivalence` (exact near ten qubits);
* **distribution** -- :func:`apedp_dyn.vqc.receipts.adjudicate` when the candidate
  declares an expected output distribution, with its abstention rule intact.

The rule this module enforces is the one the platform already applies to
hypotheses: a candidate whose equivalence cannot be *decided* is `UNRESOLVED`, and
:func:`promote` refuses an `UNRESOLVED` verdict exactly as it refuses an `INVALID`
one. "We could not check it" is not a promotion.

Four questions are answered separately, and the receipt names each one:

* `verdict` (`semantic_verdict`) -- does the candidate compute the reference operation?
* `claim_verdict` (`resource_verdict`) -- is it the improvement it is offered as?
* `evidence` (`evidence_sufficiency`) -- was anything actually checked, and did every
  check decide? An empty check set is `INSUFFICIENT`, never a pass.
* `promotable` (`promotion_decision`) -- may it be promoted? `PROMOTED` needs all
  three of the above: `VALID`, no `UNMET` claim, and `SUFFICIENT` evidence.

Folding them together is what this module did first and no longer does: an equivalent
but larger candidate used to come out `INVALID` (a false statement about the circuit),
and a candidate nothing was asked about could look like one that passed every check.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.equivalence import (
    MAX_UNITARY_QUBITS,
    CliffordEquivalenceReport,
    EquivalenceReport,
    adjudicate_clifford_equivalence,
    adjudicate_equivalence,
)
from apedp_dyn.vqc.receipts import Check, Receipt, adjudicate
from apedp_dyn.vqc.stabilizer import is_clifford_circuit

__all__ = [
    "Candidate",
    "CandidateVerdict",
    "CircuitCandidate",
    "PromotionRefused",
    "adjudicate_candidate",
    "promote",
]


@runtime_checkable
class Candidate(Protocol):
    """What admission needs of a candidate: an identity, a circuit, and claims.

    A rewrites-loop candidate carries a *reference circuit*; a synthesis candidate
    carries a *specification* and no reference at all. Both go through the same
    verdict and the same promotion policy, so the policy is stated over this shape
    rather than over the one candidate type that existed first.
    """

    candidate_id: str
    circuit: Circuit
    claims: Mapping[str, int]

    def as_dict(self) -> dict:  # pragma: no cover - structural protocol
        ...


@dataclasses.dataclass(frozen=True)
class CircuitCandidate:
    """One discovered circuit, offered against a reference circuit.

    Satisfies :class:`Candidate` structurally; the reference is what a rewrites-loop
    candidate has and a synthesis candidate does not.
    """

    candidate_id: str
    circuit: Circuit
    reference: Circuit
    source: str = "unspecified"
    #: Resource measures the candidate is offered under, e.g. ``{"two_qubit_gates": 2}``.
    claims: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: The output distribution the candidate claims, as ``{bitstring: probability}``.
    expected: Mapping[str, float] | None = None
    shots: int = 4096
    seed: int | None = 0
    metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "reference_sha256": self.reference.sha256(),
            "candidate_sha256": self.circuit.sha256(),
            "claims": dict(self.claims),
            "declares_distribution": self.expected is not None,
            "shots": self.shots,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


class PromotionRefused(RuntimeError):
    """Raised when a candidate is offered for promotion without a passing verdict."""

    def __init__(self, verdict: CandidateVerdict) -> None:
        self.verdict = verdict
        reasons = "; ".join(verdict.refused_reasons) or "no successful check"
        if verdict.evidence == "INSUFFICIENT":
            # An empty check set is never a pass: there is nothing to promote on.
            super().__init__(
                f"candidate {verdict.candidate.candidate_id!r} has insufficient "
                f"evidence: no check was run, so there is no verdict to promote on")
        elif verdict.verdict == "VALID":
            # The two refusals are different statements: "wrong" and "no cheaper".
            super().__init__(
                f"candidate {verdict.candidate.candidate_id!r} is VALID but not an "
                f"improvement ({verdict.claim_verdict}): {reasons}")
        else:
            super().__init__(
                f"candidate {verdict.candidate.candidate_id!r} is {verdict.verdict}, "
                f"not VALID: {reasons}")


@dataclasses.dataclass(frozen=True)
class CandidateVerdict:
    """The receipt a promoter has to read: both referees' evidence, and a verdict."""

    candidate: Candidate
    tier: str
    equivalence: EquivalenceReport | CliffordEquivalenceReport | None
    distribution: Receipt | None
    checks: tuple[Check, ...]
    verdict: str
    refused_reasons: tuple[str, ...]
    #: The economic claim on its own: MET/UNMET/NOT_ASSERTED. A candidate can be
    #: semantically VALID and still save nothing.
    claim_verdict: str = "NOT_ASSERTED"
    #: How much was actually checked: SUFFICIENT / UNDECIDED / INSUFFICIENT.
    #:
    #: `INSUFFICIENT` is the empty set of checks. It is never a pass: a candidate
    #: nothing was asked about is refused with that reason, not with "INVALID",
    #: because no referee said anything about it.
    evidence: str = "SUFFICIENT"

    def __post_init__(self) -> None:
        # Derive the dimension from the checks so a caller cannot set it by hand and
        # so an empty check set cannot be reported as sufficient evidence.
        object.__setattr__(self, "evidence", _evidence(self.checks))

    @property
    def promotable(self) -> bool:
        return (self.verdict == "VALID" and self.claim_verdict != "UNMET"
                and self.evidence == "SUFFICIENT")

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate.as_dict(),
            "tier": self.tier,
            "verdict": self.verdict,
            "claim_verdict": self.claim_verdict,
            "evidence": self.evidence,
            # The four questions, named separately, so no consumer can collapse them:
            # is it right, is it better, was it checked, and may it be promoted.
            "semantic_verdict": self.verdict,
            "resource_verdict": self.claim_verdict,
            "evidence_sufficiency": self.evidence,
            "promotion_decision": ("PROMOTED" if self.promotable else "REFUSED"),
            "promotable": self.promotable,
            "refused_reasons": list(self.refused_reasons),
            "equivalence": (None if self.equivalence is None
                            else self.equivalence.as_dict()),
            "distribution": (None if self.distribution is None
                             else self.distribution.as_dict()),
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def _evidence(checks: tuple[Check, ...]) -> str:
    """`SUFFICIENT` only when at least one check ran and every one was decided."""
    if not checks:
        return "INSUFFICIENT"
    if any(check.passed is None for check in checks):
        return "UNDECIDED"
    return "SUFFICIENT"


def _is_unitary(circuit: Circuit) -> bool:
    """Whether every operation is an unconditioned gate, i.e. a unitary claim exists."""
    return all(isinstance(op, Gate) and not op.condition for op in circuit.ops)


def _equivalence(candidate: CircuitCandidate, on: str, as_subroutine: bool = False
                 ) -> tuple[str, EquivalenceReport | CliffordEquivalenceReport | None,
                            Check | None]:
    """Route to the referee that can actually decide this pair.

    A circuit that measures has no unitary claim to check: the honest verdict for the
    equivalence claim is that it does not apply, and the distribution receipt decides
    instead. `None` is returned for the check in that case, so it cannot be counted
    as either a pass or a failure.

    A controlled-subroutine claim (`as_subroutine=True`) always goes to the dense tier:
    the Clifford tier's images cannot see a global phase, so it cannot decide whether a
    phase that is free standalone becomes a relative phase under control.
    """
    reference, proposed = candidate.reference, candidate.circuit
    if not (_is_unitary(reference) and _is_unitary(proposed)):
        return "distribution", None, None
    if not as_subroutine and on in ("auto", "stabilizer") \
            and is_clifford_circuit(reference) and is_clifford_circuit(proposed):
        report = adjudicate_clifford_equivalence(
            reference, proposed, claims=candidate.claims or None,
            provenance={"candidate_id": candidate.candidate_id,
                        "source": candidate.source})
        return "stabilizer", report, Check(
            name="the candidate is equivalent to its reference circuit",
            requirement="the stabilizer tier's conjugation images agree exactly",
            measured={"tier": "stabilizer", "scope": report.scope,
                      "mismatched": report.mismatched, "compared": report.compared},
            passed=(report.verdict == "VALID" if report.verdict != "UNRESOLVED" else None))

    if on in ("auto", "dense"):
        report = adjudicate_equivalence(
            reference, proposed, as_subroutine=as_subroutine,
            claims=candidate.claims or None,
            provenance={"candidate_id": candidate.candidate_id,
                        "source": candidate.source})
        decided = all(check.passed is not None for check in report.checks)
        if as_subroutine:
            requirement = (f"both unitaries materialisable at at most "
                           f"{MAX_UNITARY_QUBITS} qubits and equal as controlled "
                           f"subroutines, where a global phase is not free")
        else:
            requirement = (f"both unitaries materialisable at at most "
                           f"{MAX_UNITARY_QUBITS} qubits and equal up to one global "
                           f"phase")
        return "dense", report, Check(
            name="the candidate is equivalent to its reference circuit",
            requirement=requirement,
            measured={"tier": "dense", "verdict": report.verdict,
                      "scope": report.scope,
                      "distance": report.distance,
                      "reason": next((check.measured.get("reason") for check in
                                      report.checks if check.measured.get("reason")), None)},
            passed=(report.verdict == "VALID") if decided else None)

    if on not in ("auto", "dense", "stabilizer"):
        raise ValueError(f"on must be 'auto', 'dense' or 'stabilizer', got {on!r}")
    return "none", None, Check(
        name="the candidate is equivalent to its reference circuit",
        requirement="a referee that can decide this pair",
        measured={"reason": (
            f"the circuits are not Clifford, so the stabilizer tier cannot decide them, "
            f"and {max(reference.num_qubits, proposed.num_qubits)} qubits is above the "
            f"dense tier's {MAX_UNITARY_QUBITS}-qubit bound")},
        passed=None)


def adjudicate_candidate(candidate: CircuitCandidate, *, on: str = "auto",
                         as_subroutine: bool = False) -> CandidateVerdict:
    """Run the candidate through every referee its claims require, and seal the result.

    `on="auto"` picks the stabilizer tier when both circuits are Clifford and the dense
    tier otherwise. A pair that neither can decide gets an `UNRESOLVED` verdict and
    cannot be promoted; it is not reported as wrong, because the referees have not
    said anything about it.

    `as_subroutine=True` asks the stricter question -- may the candidate be used under
    control? -- and therefore uses the dense tier only, where a global phase can be
    compared rather than ignored.
    """
    tier, report, equivalence_check = _equivalence(candidate, on, as_subroutine)
    checks: list[Check] = [] if equivalence_check is None else [equivalence_check]
    receipt: Receipt | None = None

    if candidate.expected is not None:
        # The reference has to accept the same declared distribution; otherwise the
        # candidate is not being measured against anything it claims to replace.
        receipt = adjudicate(
            candidate.circuit, expected=dict(candidate.expected), shots=candidate.shots,
            seed=candidate.seed,
            provenance={"candidate_id": candidate.candidate_id,
                        "source": candidate.source,
                        "reference_sha256": candidate.reference.sha256()})
        checks.append(Check(
            name="the candidate reproduces the distribution it claims",
            requirement="every declared outcome within the receipt's sigma band, and "
                        "enough shots to detect the declared effect",
            measured={"distribution_verdict": receipt.verdict,
                      "shots": candidate.shots, "seed": candidate.seed},
            passed=(receipt.verdict == "VALID") if receipt.verdict != "UNRESOLVED" else None))

    if any(check.passed is False for check in checks):
        verdict = "INVALID"
    elif checks and all(check.passed is True for check in checks):
        verdict = "VALID"
    else:
        verdict = "UNRESOLVED"

    claim_verdict = "NOT_ASSERTED"
    if report is not None:
        claim_verdict = report.claim_verdict

    reasons: list[str] = []
    if not checks:
        reasons.append(
            "no claim was checked: the circuit measures, so it has no unitary claim, and "
            "no output distribution was declared")
    for check in checks:
        if check.passed is True:
            continue
        reason = check.measured.get("reason")
        if reason:
            reasons.append(f"{check.name}: {reason}")
        elif check.passed is None:
            reasons.append(f"{check.name}: undecided")
        else:
            reasons.append(f"{check.name}: failed")
    if claim_verdict == "UNMET" and report is not None and report.claim is not None:
        unmet = report.claim.measured.get("unmet") or {}
        detail = ", ".join(
            f"{measure}: claimed {row['claimed']}, measured {row['actual']}"
            for measure, row in sorted(unmet.items()))
        reasons.append(f"the candidate is equivalent but does not improve its "
                       f"resource claim ({detail})")

    return CandidateVerdict(candidate=candidate, tier=tier,
                            equivalence=report, distribution=receipt,
                            checks=tuple(checks), verdict=verdict,
                            refused_reasons=tuple(reasons),
                            claim_verdict=claim_verdict)


def promote(verdict: CandidateVerdict) -> str:
    """Return the candidate id only if it is VALID *and* meets its resource claim.

    An equivalent but non-improving candidate is refused as non-improving, not as
    wrong: `verdict.verdict` stays `VALID` and `verdict.claim_verdict` is `UNMET`.
    """
    if not verdict.promotable:
        raise PromotionRefused(verdict)
    return verdict.candidate.candidate_id
