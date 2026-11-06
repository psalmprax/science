"""A discovered circuit is promoted only when a referee has said something about it.

The cases are the ones a discovery engine actually produces: an equivalent candidate
under a resource claim, a candidate that is statistically right and unitarily wrong,
a candidate that declares nothing checkable, and a 100-qubit Clifford candidate that
only the stabilizer tier can decide.
"""
from __future__ import annotations

import json

import pytest

from apedp_dyn.vqc import Circuit, Gate, Measure
from apedp_dyn.vqc.discovery import (
    CandidateVerdict,
    CircuitCandidate,
    PromotionRefused,
    adjudicate_candidate,
    promote,
)
from apedp_dyn.vqc.stabilizer import ghz_circuit

REFERENCE_CX = Circuit(ops=(Gate("cx", (0, 1)),), num_qubits=2)
CX_FROM_CZ = Circuit(ops=(Gate("h", (1,)), Gate("cz", (0, 1)), Gate("h", (1,))), num_qubits=2)


def measured_bell() -> Circuit:
    return Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                        Measure(0, 0), Measure(1, 1)), num_qubits=2, num_bits=2)


def test_an_equivalent_candidate_is_promoted_with_its_claim_met():
    candidate = CircuitCandidate("cx-via-cz", CX_FROM_CZ, REFERENCE_CX, source="test",
                                 claims={"two_qubit_gates": 1})
    verdict = adjudicate_candidate(candidate)
    assert isinstance(verdict, CandidateVerdict)
    assert verdict.tier == "stabilizer"
    assert verdict.verdict == "VALID"
    assert promote(verdict) == "cx-via-cz"


def test_an_equivalent_but_larger_candidate_is_refused_as_non_improving():
    """Right operation, no saving: two verdicts, and only the second one fails."""
    candidate = CircuitCandidate("cx-via-cz", CX_FROM_CZ, REFERENCE_CX,
                                 claims={"operations": 1})
    verdict = adjudicate_candidate(candidate)
    assert verdict.verdict == "VALID", "the unitary is the reference's"
    assert verdict.claim_verdict == "UNMET"
    assert verdict.promotable is False
    assert "equivalent but does not improve" in verdict.refused_reasons[-1]
    with pytest.raises(PromotionRefused, match="not an improvement"):
        promote(verdict)


def test_a_phase_that_is_free_standalone_is_not_free_as_a_subroutine():
    """The stricter question is routed, decided, and reported with its scope."""
    phased = Circuit(ops=(*CX_FROM_CZ.ops, Gate("x", (1,)), Gate("z", (1,)),
                          Gate("x", (1,)), Gate("z", (1,))), num_qubits=2)
    candidate = CircuitCandidate("phased-cx", phased, REFERENCE_CX)
    assert adjudicate_candidate(candidate).verdict == "VALID"

    controlled = adjudicate_candidate(candidate, as_subroutine=True)
    assert controlled.tier == "dense"
    assert controlled.equivalence.scope == "controlled"
    assert controlled.verdict == "INVALID"
    with pytest.raises(PromotionRefused, match="not VALID"):
        promote(controlled)

    # Above the dense bound the controlled question is undecided, not guessed: the
    # stabilizer tier's images cannot see a global phase at all.
    chain = ghz_circuit(100, measure=False)
    large = Circuit(ops=(*chain.ops, Gate("x", (5,)), Gate("z", (5,)),
                         Gate("x", (5,)), Gate("z", (5,))), num_qubits=100)
    above = adjudicate_candidate(CircuitCandidate("phased-ghz", large, chain),
                                 as_subroutine=True)
    assert above.tier == "dense"
    assert above.verdict == "UNRESOLVED"


def test_a_relative_phase_candidate_is_refused_even_though_every_probability_matches():
    """The failure mode the whole equivalence layer exists for."""
    candidate = CircuitCandidate(
        "cx-plus-z", Circuit(ops=(Gate("cx", (0, 1)), Gate("z", (1,))), num_qubits=2),
        REFERENCE_CX)
    verdict = adjudicate_candidate(candidate)
    assert verdict.verdict == "INVALID"
    assert verdict.equivalence.mismatched > 0
    with pytest.raises(PromotionRefused):
        promote(verdict)


def test_a_measured_candidate_is_decided_by_its_distribution_receipt():
    circuit = measured_bell()
    candidate = CircuitCandidate("bell", circuit, circuit, source="test",
                                 expected={"00": 0.5, "11": 0.5}, shots=2000, seed=1)
    verdict = adjudicate_candidate(candidate)
    assert verdict.tier == "distribution"
    assert verdict.equivalence is None
    assert verdict.distribution is not None and verdict.distribution.verdict == "VALID"
    assert verdict.verdict == "VALID"
    assert promote(verdict) == "bell"


def test_a_measured_candidate_with_the_wrong_distribution_is_refused():
    circuit = measured_bell()
    # Declaring a certainty that does not happen is a false claim, not a weak one.
    candidate = CircuitCandidate("bell", circuit, circuit, expected={"00": 1.0},
                                 shots=300, seed=1)
    verdict = adjudicate_candidate(candidate)
    assert verdict.verdict == "INVALID"
    assert verdict.checks[-1].measured["distribution_verdict"] == "INVALID"
    with pytest.raises(PromotionRefused):
        promote(verdict)


def test_an_underpowered_distribution_claim_does_not_become_a_pass():
    circuit = measured_bell()
    candidate = CircuitCandidate("bell", circuit, circuit,
                                 expected={"00": 0.5, "11": 0.5}, shots=300, seed=1)
    verdict = adjudicate_candidate(candidate)
    assert verdict.verdict == "UNRESOLVED"
    assert verdict.distribution.verdict == "UNRESOLVED"
    with pytest.raises(PromotionRefused):
        promote(verdict)


def test_a_candidate_that_declares_nothing_checkable_is_insufficient_evidence():
    """The mandatory rule: an empty check set is not a pass.

    Nothing was asked, so nothing was answered. The verdict is `UNRESOLVED` and the
    evidence dimension says `INSUFFICIENT`; `promote` refuses with that reason rather
    than pretending an empty collection of checks is a successful verification.
    """
    candidate = CircuitCandidate("measured-no-claim", measured_bell(), measured_bell())
    verdict = adjudicate_candidate(candidate)
    assert verdict.verdict == "UNRESOLVED"
    assert not verdict.checks
    assert verdict.evidence == "INSUFFICIENT"
    assert verdict.promotable is False
    assert "no claim was checked" in verdict.refused_reasons[0]
    with pytest.raises(PromotionRefused, match="insufficient evidence"):
        promote(verdict)
    payload = json.loads(verdict.to_json())
    assert payload["evidence_sufficiency"] == "INSUFFICIENT"
    assert payload["promotion_decision"] == "REFUSED"


def test_the_four_adjudication_questions_are_separate_fields():
    """Semantics, economics, evidence and the decision are four values, not one."""
    met = adjudicate_candidate(CircuitCandidate("cx-via-cz", CX_FROM_CZ, REFERENCE_CX,
                                                claims={"two_qubit_gates": 1}))
    payload = json.loads(met.to_json())
    assert payload["semantic_verdict"] == "VALID"
    assert payload["resource_verdict"] == "MET"
    assert payload["evidence_sufficiency"] == "SUFFICIENT"
    assert payload["promotion_decision"] == "PROMOTED"

    # Semantically right, not cheaper: one field changes, and it is not the verdict.
    larger = adjudicate_candidate(CircuitCandidate("cx-via-cz", CX_FROM_CZ, REFERENCE_CX,
                                                   claims={"operations": 1}))
    payload = json.loads(larger.to_json())
    assert payload["semantic_verdict"] == "VALID"
    assert payload["resource_verdict"] == "UNMET"
    assert payload["evidence_sufficiency"] == "SUFFICIENT"
    assert payload["promotion_decision"] == "REFUSED"

    # A referee that abstains is UNDECIDED evidence, not a pass and not a failure.
    uncheckable = adjudicate_candidate(CircuitCandidate(
        "uncheckable",
        Circuit(ops=tuple(Gate("t", (qubit,)) for qubit in range(19)), num_qubits=20),
        Circuit(ops=tuple(Gate("t", (qubit,)) for qubit in range(20)), num_qubits=20)))
    assert uncheckable.evidence == "UNDECIDED"
    assert uncheckable.promotable is False


def test_a_clifford_candidate_above_the_dense_bound_is_decided():
    chain = ghz_circuit(100, measure=False)
    star = Circuit(ops=(Gate("h", (0,)),
                        *[Gate("cx", (0, qubit)) for qubit in range(1, 100)]),
                   num_qubits=100)
    refused = adjudicate_candidate(CircuitCandidate("star", star, chain))
    assert refused.tier == "stabilizer"
    assert refused.verdict == "INVALID"

    # A global phase is not an error, at any size.
    phased = Circuit(ops=(*chain.ops, Gate("x", (5,)), Gate("z", (5,)),
                          Gate("x", (5,)), Gate("z", (5,))), num_qubits=100)
    accepted = adjudicate_candidate(CircuitCandidate("phased", phased, chain))
    assert accepted.verdict == "VALID"
    assert accepted.equivalence.compared == 200
    assert promote(accepted) == "phased"


def test_a_non_clifford_candidate_above_the_dense_bound_is_unresolved_not_wrong():
    reference = Circuit(ops=tuple(Gate("t", (qubit,)) for qubit in range(20)),
                        num_qubits=20)
    candidate = Circuit(ops=(*(Gate("t", (qubit,)) for qubit in range(19)),
                             Gate("s", (19,))), num_qubits=20)
    verdict = adjudicate_candidate(CircuitCandidate("uncheckable", candidate, reference))
    assert verdict.tier == "dense"
    assert verdict.verdict == "UNRESOLVED"
    # The dense referee's own report is kept as evidence: it says it could not decide.
    assert verdict.equivalence.verdict == "UNRESOLVED"
    assert verdict.equivalence.distance is None
    assert "this checker stops at" in verdict.refused_reasons[0]
    with pytest.raises(PromotionRefused):
        promote(verdict)


def test_the_route_can_be_forced():
    with pytest.raises(ValueError, match="on must be"):
        adjudicate_candidate(CircuitCandidate("x", CX_FROM_CZ, REFERENCE_CX), on="guess")
    forced = adjudicate_candidate(CircuitCandidate("x", CX_FROM_CZ, REFERENCE_CX),
                                  on="dense")
    assert forced.tier == "dense"
    assert forced.verdict == "VALID"
    assert forced.equivalence.global_phase == pytest.approx(0.0)


def test_the_verdict_is_a_receipt():
    verdict = adjudicate_candidate(CircuitCandidate("bell", measured_bell(), measured_bell(),
                                                    expected={"00": 0.5, "11": 0.5},
                                                    shots=2000, seed=2, source="engine"))
    payload = json.loads(verdict.to_json())
    assert payload["candidate"]["source"] == "engine"
    assert payload["candidate"]["declares_distribution"] is True
    assert payload["verdict"] == "VALID"
    assert payload["promotable"] is True
    assert payload["distribution"]["verdict"] == "VALID"
    assert payload["tier"] == "distribution"
    assert verdict.as_dict() == json.loads(verdict.to_json())
