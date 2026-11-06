"""A corpus of circuit pairs built so that two equivalence referees disagree.

Status: experimental

An equivalence comparison on pairs where both tools answer the same obvious thing
measures nothing. The record that preceded this module said exactly that: "the
comparison that would matter is a corpus where the two disagree, and it does not
exist yet". This is that corpus, and its design rule is:

    each pair's relation is fixed by how it was *built*, not by either referee.

A pair carries the construction that establishes its relation -- ``X Y Z = i I``,
``RZ(2*pi) = -I``, ``SWAP = CX(a,b) CX(b,a) CX(a,b)``, ``Z_0 Z_1`` stabilising a GHZ,
``U_c = RX(theta) U_r`` with ``RX(theta)`` not a scalar -- so the truth is an
algebraic identity in the generator, and the verdicts are measurements against it.

Where the two referees differ, they differ along three axes that this corpus
isolates rather than averages:

**Tolerance.** An exactly inequivalent pair whose difference is smaller than a
referee's threshold gets a different answer from a referee with a looser one. The
ladder family varies one angle from ``1e-13`` to ``1e-3`` and records both tools'
edges; the ``U_c = RX(theta) U_r`` construction is the statement that the pair is
*not* exactly equivalent at any of them, so a ``VALID`` at ``1e-9`` is a threshold
decision and is reported as one.

**Scope.** ``diag(I, U)`` is not ``diag(I, e^(i phi) U)``. A global phase is free for
a standalone operation and is a relative phase for a controlled one, and a pair of
circuits that prepare the same state can be different unitaries. A referee that
decides only one of those questions has not made a mistake when it answers
differently from one that decides another; it has answered a different question. The
corpus records every scope our referee offers beside the pair, so the reader can see
that the apparent disagreement is a scope difference and not a contradiction.

**Abstention.** Above the dense bound a non-Clifford pair is ``UNRESOLVED`` here and
a decision-diagram checker decides it; near a threshold a third-party checker can
report ``no_information`` where a dense matrix is exact. An abstention is recorded as
an abstention on both sides, never as a win or a loss.

What this module does *not* decide is which tool is right. It decides what each tool
answered, on which question, against a truth that was fixed before either ran.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.equivalence import (
    MAX_UNITARY_QUBITS,
    adjudicate_clifford_equivalence,
    adjudicate_equivalence,
)

__all__ = [
    "EQUIVALENCE_TOLERANCE",
    "SCOPES",
    "EquivalencePair",
    "ScopeVerdict",
    "adjudicate_pair",
    "classify",
    "equivalence_pairs",
    "summarise",
]

#: This package's equivalence tolerance, on the per-amplitude Frobenius distance to
#: the closest global-phase multiple. It is the referee's default, restated here so
#: the corpus's `expected` verdicts are read against a named number rather than an
#: implicit one. QCEC's threshold is not this number, which is the point of the
#: tolerance family.
EQUIVALENCE_TOLERANCE = 1e-9

#: The three questions our referee can be asked about a pair. QCEC decides one.
SCOPES = ("standalone", "controlled", "state")


@dataclasses.dataclass(frozen=True)
class ScopeVerdict:
    """What this package answered about one pair, on one question."""

    scope: str
    verdict: str
    tier: str
    decides: str
    seconds: float
    distance: float | None = None
    entry_deviation: float | None = None
    tolerance: float | None = None
    checks_failed: tuple[str, ...] = ()
    #: Why the question could not be decided, for ``UNRESOLVED``.
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "verdict": self.verdict,
            "tier": self.tier,
            "decides": self.decides,
            "distance": self.distance,
            "entry_deviation": self.entry_deviation,
            "tolerance": self.tolerance,
            "checks_failed": list(self.checks_failed),
            "seconds": self.seconds,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class EquivalencePair:
    """Two circuits, how they were built to relate, and what a referee should answer.

    `expected` is keyed by scope and is part of the corpus, not of a tool: it is what
    the construction says a referee holding `EQUIVALENCE_TOLERANCE` must answer, which
    is what makes the corpus a test of our referee as well as a comparison with the
    other one.
    """

    label: str
    family: str
    question: str
    relation: str
    construction: str
    reference: Circuit
    candidate: Circuit
    expected: Mapping[str, str]

    @property
    def qubits(self) -> int:
        return self.reference.num_qubits

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family": self.family,
            "question": self.question,
            "relation": self.relation,
            "construction": self.construction,
            "qubits": self.qubits,
            "expected": dict(self.expected),
            "reference_sha256": self.reference.sha256(),
            "candidate_sha256": self.candidate.sha256(),
            "reference_operations": int(self.reference.resources()["operations"]),
            "candidate_operations": int(self.candidate.resources()["operations"]),
        }


def _ghz(num_qubits: int) -> tuple[Gate, ...]:
    """A GHZ ladder: H on the first wire and a CX along every edge."""
    return (Gate("h", (0,)), *[Gate("cx", (index, index + 1))
                               for index in range(num_qubits - 1)])


def _pairs() -> tuple[EquivalencePair, ...]:
    ghz3 = Circuit(num_qubits=3, ops=_ghz(3))
    pairs: list[EquivalencePair] = [
        EquivalencePair(
            label="rewrite-cancelled-pair",
            family="rewrite",
            question="standalone",
            relation="equal",
            construction=("two X gates on the same wire in the middle of the schedule: "
                          "X X = I, so the candidate is the reference unitary exactly"),
            reference=ghz3,
            candidate=Circuit(num_qubits=3, ops=(Gate("h", (0,)), Gate("x", (0,)),
                                                 Gate("x", (0,)), Gate("cx", (0, 1)),
                                                 Gate("cx", (1, 2)))),
            expected={"standalone": "VALID", "controlled": "VALID"},
        ),
        EquivalencePair(
            label="rewrite-independent-blocks",
            family="rewrite",
            question="standalone",
            relation="equal",
            construction=("the two preparation blocks act on disjoint wires "
                          "(0,1) and (2,3) and therefore commute, so the reordered "
                          "schedule is the same unitary"),
            reference=Circuit(num_qubits=4, ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                                                 Gate("h", (2,)), Gate("cx", (2, 3)),
                                                 Gate("cx", (1, 2)))),
            candidate=Circuit(num_qubits=4, ops=(Gate("h", (2,)), Gate("cx", (2, 3)),
                                                 Gate("h", (0,)), Gate("cx", (0, 1)),
                                                 Gate("cx", (1, 2)))),
            expected={"standalone": "VALID", "controlled": "VALID"},
        ),
        EquivalencePair(
            label="rewrite-swap-decomposition",
            family="rewrite",
            question="standalone",
            relation="equal",
            construction=("SWAP(a,b) = CX(a,b) CX(b,a) CX(a,b), so a circuit using the "
                          "gate and one using its decomposition are the same unitary; "
                          "the pair also checks that the other tool receives `swap` as "
                          "something it can decide"),
            reference=Circuit(num_qubits=3, ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                                                 Gate("swap", (1, 2)))),
            candidate=Circuit(num_qubits=3, ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                                                 Gate("cx", (1, 2)), Gate("cx", (2, 1)),
                                                 Gate("cx", (1, 2)))),
            expected={"standalone": "VALID", "controlled": "VALID"},
        ),
        EquivalencePair(
            label="phase-xyz-is-i",
            family="global_phase",
            question="standalone",
            relation="equal_up_to_global_phase",
            construction=("X Y Z = i I on one qubit, so the candidate is the identity "
                          "times i: equal standalone, and not equal under control, "
                          "where diag(I, i I) is a relative phase"),
            reference=Circuit(num_qubits=1, ops=()),
            candidate=Circuit(num_qubits=1, ops=(Gate("x", (0,)), Gate("y", (0,)),
                                                 Gate("z", (0,)))),
            expected={"standalone": "VALID", "controlled": "INVALID"},
        ),
        EquivalencePair(
            label="phase-rz-2pi-is-minus-one",
            family="global_phase",
            question="standalone",
            relation="equal_up_to_global_phase",
            construction=("RZ(2*pi) = -I, so appending it multiplies the unitary by -1: "
                          "equal standalone, a relative phase under control"),
            reference=Circuit(num_qubits=2, ops=(Gate("h", (0,)), Gate("cx", (0, 1)))),
            candidate=Circuit(num_qubits=2, ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                                                 Gate("rz", (0,), (2.0 * math.pi,)))),
            expected={"standalone": "VALID", "controlled": "INVALID"},
        ),
        EquivalencePair(
            label="relative-s-versus-sdg",
            family="relative_phase",
            question="standalone",
            relation="unequal",
            construction=("S and Sdg differ by Z on the same wire, which is a relative "
                          "phase and not a global one: the two unitaries are different "
                          "at the same distance in every scope"),
            reference=Circuit(num_qubits=3, ops=(*_ghz(3), Gate("s", (0,)))),
            candidate=Circuit(num_qubits=3, ops=(*_ghz(3), Gate("sdg", (0,)))),
            expected={"standalone": "INVALID", "controlled": "INVALID",
                      "state": "INVALID"},
        ),
        EquivalencePair(
            label="relative-diagonal-cz",
            family="relative_phase",
            question="standalone",
            relation="unequal",
            construction=("CZ is diagonal, so both circuits are diagonal and every "
                          "computational-basis probability is identical; the unitaries "
                          "differ only in phase, which is what a sampling referee "
                          "cannot see and a unitary referee can"),
            reference=ghz3,
            candidate=Circuit(num_qubits=3, ops=(*_ghz(3), Gate("cz", (0, 1)))),
            expected={"standalone": "INVALID", "controlled": "INVALID",
                      "state": "INVALID"},
        ),
        EquivalencePair(
            label="state-ghz-stabiliser",
            family="state",
            question="state",
            relation="same_prepared_state",
            construction=("Z_0 Z_1 stabilises the GHZ state, so appending it leaves the "
                          "prepared state unchanged while changing the unitary: equal "
                          "as states, different as operations"),
            reference=ghz3,
            candidate=Circuit(num_qubits=3, ops=(*_ghz(3), Gate("z", (0,)),
                                                       Gate("z", (1,)))),
            expected={"state": "VALID", "standalone": "INVALID",
                      "controlled": "INVALID"},
        ),
    ]

    # The tolerance ladder: one angle, seven orders of magnitude. Every pair is
    # exactly inequivalent by construction (`U_c = RX(theta) U_r` with a non-scalar
    # rotation), so a `VALID` anywhere on it is a threshold decision, not an identity.
    for exponent, theta, expected in (
            (13, 1e-13, "VALID"),
            (9, 1e-9, "VALID"),
            (8, 1e-8, "INVALID"),
            (7, 1e-7, "INVALID"),
            (6, 1e-6, "INVALID"),
            (5, 1e-5, "INVALID"),
            (3, 1e-3, "INVALID")):
        pairs.append(EquivalencePair(
            label=f"tolerance-1e-{exponent:02d}",
            family="tolerance",
            question="standalone",
            relation="unequal",
            construction=(f"the candidate is RX({theta:g}) applied after the reference, "
                          f"and RX({theta:g}) is not a scalar multiple of the identity, "
                          f"so the two unitaries are exactly different; the per-amplitude "
                          f"distance is {theta / 2:g}, which is the number each "
                          f"referee's threshold is compared against"),
            reference=Circuit(num_qubits=1, ops=(Gate("h", (0,)),)),
            candidate=Circuit(num_qubits=1, ops=(Gate("h", (0,)),
                                                 Gate("rx", (0,), (theta,)))),
            expected={"standalone": expected},
        ))

    # Above the dense bound. A Clifford pair is still decided here (stabilizer tier);
    # a non-Clifford pair is not, and the other tool decides both.
    wide = Circuit(num_qubits=12, ops=_ghz(12))
    pairs.extend([
        EquivalencePair(
            label="wide-clifford-rewrite",
            family="wide",
            question="standalone",
            relation="equal",
            construction=("12 qubits, beyond the 10-qubit dense bound, with a cancelling "
                          "X pair: the stabilizer tier decides the conjugation action "
                          "exactly, so this is the row where the bound is not an "
                          "abstention"),
            reference=wide,
            candidate=Circuit(num_qubits=12, ops=(*_ghz(12), Gate("x", (0,)),
                                                          Gate("x", (0,)))),
            expected={"standalone": "VALID"},
        ),
        EquivalencePair(
            label="wide-non-clifford-cancelled",
            family="wide",
            question="standalone",
            relation="equal",
            construction=("12 qubits with T Tdg appended: exactly equal, and outside "
                          "both of this package's tiers (not dense at 12 qubits, not "
                          "Clifford because of the T)"),
            reference=wide,
            candidate=Circuit(num_qubits=12, ops=(*_ghz(12), Gate("t", (3,)),
                                                          Gate("tdg", (3,)))),
            expected={"standalone": "UNRESOLVED"},
        ),
        EquivalencePair(
            label="wide-non-clifford-differing",
            family="wide",
            question="standalone",
            relation="unequal",
            construction=("12 qubits with T on one side and S on the other: a relative "
                          "phase this package's tiers do not reach at this width"),
            reference=Circuit(num_qubits=12, ops=(*_ghz(12), Gate("t", (3,)))),
            candidate=Circuit(num_qubits=12, ops=(*_ghz(12), Gate("s", (3,)))),
            expected={"standalone": "UNRESOLVED"},
        ),
    ])
    return tuple(pairs)


def equivalence_pairs() -> tuple[EquivalencePair, ...]:
    """The corpus, rebuilt from the constructions on every call."""
    return _pairs()


# --------------------------------------------------------------- our verdicts
def _refusal(report) -> str | None:
    """The reason a stabilizer report refused, or None if it decided.

    `adjudicate_clifford_equivalence` reports a non-Clifford circuit as a failed
    check and a failed check as ``INVALID``, which would be a verdict of inequality
    where the tier has no verdict at all. The refusal is therefore read off the
    admissibility check by name rather than off the verdict.
    """
    for check in report.checks:
        if check.passed is False and "Clifford unitary circuit" in check.name:
            return str(check.measured.get("reason", "not a Clifford unitary circuit"))
    return None


def _from_dense(scope: str, report, seconds: float,
                tolerance: float) -> ScopeVerdict:
    reason = None
    if report.verdict == "UNRESOLVED":
        reason = next((str(check.measured.get("reason"))
                       for check in report.checks if check.passed is None), None)
    return ScopeVerdict(
        scope=scope,
        verdict=report.verdict,
        tier="dense",
        decides=report.as_dict()["decides"],
        seconds=seconds,
        distance=report.distance,
        entry_deviation=report.entry_deviation,
        tolerance=tolerance,
        checks_failed=report.failed(),
        reason=reason,
    )


def _from_clifford(scope: str, report, seconds: float) -> ScopeVerdict:
    refusal = _refusal(report)
    return ScopeVerdict(
        scope=scope,
        verdict="UNRESOLVED" if refusal else report.verdict,
        tier="stabilizer",
        decides=report.as_dict()["decides"],
        seconds=seconds,
        checks_failed=report.failed(),
        reason=refusal,
    )


def adjudicate_pair(pair: EquivalencePair, *,
                    tolerance: float = EQUIVALENCE_TOLERANCE,
                    ) -> dict[str, ScopeVerdict]:
    """Ask this package every question it can be asked about `pair`.

    All three scopes are attempted for every pair, and the ones the tiers cannot
    reach come back ``UNRESOLVED`` with the reason instead of being omitted: an
    abstention that is not in the record is an abstention a reader will assume away.
    """
    if pair.reference.num_qubits != pair.candidate.num_qubits:
        # Not a pair this corpus builds, and not a question a width-bounded matrix
        # can answer either.
        return {scope: ScopeVerdict(scope=scope, verdict="UNRESOLVED", tier="none",
                                    decides="nothing: different widths",
                                    seconds=0.0, reason="the two circuits act on "
                                                        "different widths")
                for scope in SCOPES}

    dense = pair.qubits <= MAX_UNITARY_QUBITS

    start = time.perf_counter()
    if dense:
        standalone = _from_dense("standalone", adjudicate_equivalence(
            pair.reference, pair.candidate, tolerance=tolerance),
            time.perf_counter() - start, tolerance)
    else:
        standalone = _from_clifford("standalone", adjudicate_clifford_equivalence(
            pair.reference, pair.candidate, on="unitary"), time.perf_counter() - start)

    start = time.perf_counter()
    if dense:
        controlled = _from_dense("controlled", adjudicate_equivalence(
            pair.reference, pair.candidate, tolerance=tolerance, as_subroutine=True),
            time.perf_counter() - start, tolerance)
    else:
        # `controlled_lift` needs 2**(n+1); the stabilizer tier's images cannot see a
        # global phase, which is exactly what the controlled question is about.
        controlled = ScopeVerdict(
            scope="controlled", verdict="UNRESOLVED", tier="dense",
            decides=("diag(I, U_c) = e^(i*phi) diag(I, U_r)"),
            seconds=0.0,
            reason=(f"the controlled comparison lifts a 2**n matrix to 2**(n+1) and "
                    f"{pair.qubits} qubits is above the {MAX_UNITARY_QUBITS}-qubit "
                    f"bound; the stabilizer tier's images cannot see a global phase"))

    start = time.perf_counter()
    state = _from_clifford("state", adjudicate_clifford_equivalence(
        pair.reference, pair.candidate, on="state"), time.perf_counter() - start)

    return {"standalone": standalone, "controlled": controlled, "state": state}


# ------------------------------------------------------------------ comparison
def classify(pair: EquivalencePair, ours: Mapping[str, ScopeVerdict], *,
             foreign_verdict: str | None,
             foreign_criteria: tuple[str, ...] = ()) -> dict[str, Any]:
    """Name what the two answers are, without deciding who is right.

    `foreign_verdict` is the other tool's normalised answer: ``equivalent``,
    ``equivalent_up_to_global_phase``, ``not_equivalent``, ``no_information``,
    ``unstable`` (its own criteria differed between identical runs) or None when it
    could not be run at all.
    """
    asked = ours[pair.question]
    decided_by_foreign = foreign_verdict in (
        "equivalent", "equivalent_up_to_global_phase", "not_equivalent")

    if foreign_verdict is None:
        return {
            "outcome": "UNAVAILABLE",
            "axis": "environment",
            "explanation": ("the other tool could not be run here, so this pair is an "
                            "abstention about the environment"),
            "compared": False,
        }

    if pair.question == "state":
        return {
            "outcome": "SCOPE",
            "axis": "scope",
            "explanation": (
                "the pair is equal as *prepared states* and different as unitaries; "
                "this tool decides unitaries only, so its answer is about a different "
                "question. This package's unitary verdict is "
                f"{ours['standalone'].verdict} and can be compared with it"),
            "compared": False,
        }

    if asked.verdict == "UNRESOLVED" and foreign_verdict == "no_information":
        return {
            "outcome": "BOTH_ABSTAIN",
            "axis": "abstention",
            "explanation": ("neither tool decided: this package's tier cannot reach "
                            f"this pair ({asked.reason}) and the other tool returned "
                            "no_information"),
            "compared": False,
        }
    if asked.verdict == "UNRESOLVED":
        return {
            "outcome": "WE_ABSTAIN",
            "axis": "tier_boundary",
            "explanation": (f"this package cannot reach the pair ({asked.reason}) and "
                            f"the other tool answers {foreign_verdict}; its answer is "
                            "recorded and is not counted as a win over an abstention"),
            "compared": False,
        }
    if foreign_verdict == "no_information":
        return {
            "outcome": "QCEC_ABSTAINS",
            "axis": "qc_abstains",
            "explanation": (f"this package decides {asked.verdict}"
                            + (f" at distance {asked.distance:.3g}"
                               if asked.distance is not None else "")
                            + " and the other tool returns no_information on the same "
                              "pair; it did not disagree, it declined"),
            "compared": False,
        }
    if foreign_verdict == "unstable":
        return {
            "outcome": "UNSTABLE",
            "axis": "qc_threshold",
            "explanation": (f"identical runs of the other tool returned "
                            f"{sorted(set(foreign_criteria))} on the same pair, so its "
                            f"answer is not a function of the input; this package "
                            f"decides {asked.verdict} deterministically"
                            + (f" at distance {asked.distance:.3g}"
                               if asked.distance is not None else "")),
            "compared": False,
        }

    ours_valid = asked.verdict == "VALID"
    theirs_valid = foreign_verdict in ("equivalent", "equivalent_up_to_global_phase")
    if ours_valid == theirs_valid:
        return {
            "outcome": "AGREE",
            "axis": None,
            "explanation": (f"both answer {'equality' if ours_valid else 'inequality'}"
                            + (" (the other tool names the global phase explicitly)"
                               if foreign_verdict == "equivalent_up_to_global_phase"
                               else "")),
            "compared": True,
        }
    if pair.relation == "unequal":
        # The construction says the pair is exactly inequivalent; the two verdicts
        # differ because the thresholds do, not because the arithmetic does.
        return {
            "outcome": "DIFFER_BY_THRESHOLD",
            "axis": "tolerance",
            "explanation": (
                "the pair is exactly inequivalent by construction and the two tools "
                f"apply different thresholds: this package says {asked.verdict} at "
                f"tolerance {asked.tolerance:g}"
                + (f" (measured distance {asked.distance:.3g})"
                   if asked.distance is not None else "")
                + f", the other tool says {foreign_verdict}"
                  + (" with its own default numerical tolerance, which is not "
                     "the same quantity" if decided_by_foreign else "")),
            "compared": True,
        }
    return {
        "outcome": "CONTRADICTION",
        "axis": "unexplained",
        "explanation": (
            f"the construction says {pair.relation!r}, this package says "
            f"{asked.verdict} and the other tool says {foreign_verdict}; no threshold "
            "or scope difference explains that"),
        "compared": True,
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts in a record, recomputable from its rows.

    `expected_met` re-checks every pair's declared expectation against the verdict
    this package actually produced, so the summary cannot agree with itself while
    disagreeing with the corpus.
    """
    outcomes: dict[str, int] = {}
    axes: dict[str, int] = {}
    families: dict[str, int] = {}
    contradictions: list[str] = []
    unstable: list[str] = []
    for row in rows:
        outcome = row["agreement"]["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        axis = row["agreement"]["axis"]
        if axis is not None:
            axes[axis] = axes.get(axis, 0) + 1
        families[row["family"]] = families.get(row["family"], 0) + 1
        if outcome == "CONTRADICTION":
            contradictions.append(row["label"])
        if outcome == "UNSTABLE":
            unstable.append(row["label"])

    expected_met = []
    for row in rows:
        for scope, expected in row["expected"].items():
            if row["ours"][scope]["verdict"] != expected:
                expected_met.append({"label": row["label"], "scope": scope,
                                     "expected": expected,
                                     "verdict": row["ours"][scope]["verdict"]})

    def _count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    return {
        "pairs": len(rows),
        "families": families,
        "outcomes": outcomes,
        "axes": axes,
        "our_decided": _count(lambda row: row["ours"][row["question"]]["verdict"]
                              in ("VALID", "INVALID")),
        "our_abstained": _count(lambda row: row["ours"][row["question"]]["verdict"]
                                == "UNRESOLVED"),
        "foreign_decided": _count(lambda row: row["qcec"]["status"] == "compared"
                                  and row["qcec"]["verdict"] in (
                                      "equivalent", "equivalent_up_to_global_phase",
                                      "not_equivalent")),
        "foreign_abstained": _count(lambda row: row["qcec"]["status"] == "compared"
                                    and row["qcec"]["verdict"] in (
                                        "no_information", "unstable")),
        "foreign_unstable": unstable,
        "contradictions": contradictions,
        "expected_met": expected_met,
    }
