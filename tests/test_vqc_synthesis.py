"""Specification-driven synthesis: constructions, refusals and the shared policy.

The experiment these tests pin is the review's first: give the platform a
mathematical target and *no reference circuit*, and ask it to construct a circuit
and decide whether the construction is right.

What is pinned:

* the specification is checkable in itself -- a target that is not a pure state, or
  one with a support that does not sum to one, is refused before any search runs;
* the two readings of a specification (the tableau at any width, the projected state
  vector at small width) agree, because both referees rest on that;
* every construction family actually reproduces the target, and a wrong construction
  is refused with the stabilizer that failed named;
* the dense tier decides a non-Clifford construction, which is the boundary
  Experiment C has to cross, and says `UNRESOLVED` rather than guessing above it;
* admission is the *same* policy as the rewrites loop, including the claim rule
  (which is pinned against the equivalence referee's own implementation) and the
  rule that an uncheckable candidate is refused rather than promoted.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apedp_dyn.vqc.circuit import Circuit, Gate, Measure
from apedp_dyn.vqc.discovery import CandidateVerdict, PromotionRefused, promote
from apedp_dyn.vqc.equivalence import _claim_check as equivalence_claim_check
from apedp_dyn.vqc.stabilizer import StabilizerState, is_clifford_circuit
from apedp_dyn.vqc.synthesis import (
    DENSE_SPEC_LIMIT,
    DENSE_TOLERANCE,
    FAMILIES,
    REFUSAL_FAMILIES,
    SpecificationCandidate,
    SynthesisError,
    TargetSpecification,
    _phase_free_distance,
    _prepare_dense,
    _specification_order,
    adjudicate_specification,
    adjudicate_specification_candidate,
    build,
    cluster_state_target,
    default_families,
    dense_oracle_check,
    dense_tier_cost,
    entangling_depth_lower_bound,
    family_targets,
    ghz_target,
    graph_of_state,
    irregular_graph_target,
    lattice_edges,
    referee_agreement,
    ring_graph_target,
    seeded_graph_target,
    specification_state_vector,
    synthesize,
)
from apedp_dyn.vqc.synthesis import (
    _claim_check as synthesis_claim_check,
)


def _target_for(family: str, num_qubits: int = 6):
    """The target a family is declared for, at a size it can build."""
    name = family_targets(family)[0]
    if name == "ghz":
        return ghz_target(num_qubits)
    if name == "ring_graph":
        return ring_graph_target(num_qubits)
    if name == "cluster_2d":
        return cluster_state_target(2, num_qubits // 2)
    if name == "irregular_graph":
        return seeded_graph_target(num_qubits, seed=5)
    raise AssertionError(f"no target size chosen for {name!r}")


# ------------------------------------------------------------------ specifications
def test_a_specification_that_is_not_a_pure_state_is_refused():
    # `Z_0 Z_0` twice is rank-deficient: the group has 2 elements, not 4.
    with pytest.raises(Exception, match=r"not pure|rank"):
        TargetSpecification(name="bad", num_qubits=2,
                            stabilizers=(((3, 3), 0), ((3, 3), 0)))
    # `X_0` and `Z_0` anticommute: there is no simultaneous eigenspace.
    with pytest.raises(Exception, match="anticommute"):
        TargetSpecification(name="bad", num_qubits=2,
                            stabilizers=(((1, 0), 0), ((3, 0), 0)))
    # A Hermitian stabilizer cannot carry a phase of i.
    with pytest.raises(SynthesisError, match="imaginary"):
        TargetSpecification(name="bad", num_qubits=1, stabilizers=(((1,), 1),))
    # The wrong number of generators is not a pure state.
    with pytest.raises(SynthesisError, match="needs 2 stabilizers"):
        TargetSpecification(name="bad", num_qubits=2, stabilizers=(((1, 1), 0),))
    # A declared support must be a distribution.
    with pytest.raises(SynthesisError, match="sums to"):
        TargetSpecification(name="bad", num_qubits=2, stabilizers=(((1, 1), 0), ((3, 3), 0)),
                            support={(False, False): 0.5})


def test_the_two_readings_of_a_specification_agree():
    """The tableau and the projected vector are independent routes to the same state."""
    for num_qubits in (2, 3, 5, 8, DENSE_SPEC_LIMIT):
        evidence = dense_oracle_check(ghz_target(num_qubits))
        assert evidence["stabilizer_group_agrees"] is True
        assert evidence["support_agrees"] is True
        assert evidence["non_zero_amplitudes"] == 2
    for num_qubits in (3, 4, 6):
        evidence = dense_oracle_check(ring_graph_target(num_qubits))
        assert evidence["stabilizer_group_agrees"] is True
        # A ring graph state has 2**n amplitudes, all of them non-zero.
        assert evidence["non_zero_amplitudes"] == 2 ** num_qubits


def test_the_projected_state_really_is_the_ghz_state():
    vector = specification_state_vector(ghz_target(4))
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = expected[15] = 1.0 / math.sqrt(2)
    assert np.allclose(vector, expected, atol=1e-15)


def test_a_specification_beyond_the_dense_bound_refuses_to_materialise():
    with pytest.raises(SynthesisError, match="dense bound"):
        specification_state_vector(ghz_target(DENSE_SPEC_LIMIT + 1))


# ------------------------------------------------ the dense tier's measured cost
def test_the_cost_probe_measures_the_same_decision_the_referee_makes():
    """A price is only meaningful if it is the price of the real arithmetic."""
    target = ghz_target(5)
    circuit = _non_clifford_ghz(5)
    report = adjudicate_specification(circuit, target, tier="dense")
    referee_distance = next(check for check in report.checks
                            if check.measured.get("tier") == "dense").measured["distance"]
    cost = dense_tier_cost(circuit, target)
    assert cost.within_tolerance is True
    assert cost.distance == referee_distance
    assert cost.seconds > 0.0
    assert cost.peak_traced_bytes > 0
    assert cost.amplitudes == 2 ** 5
    # The dominant term is the dense Pauli operator, not the state vector.
    assert cost.pauli_operator_bytes == 16 * 4 ** 5
    assert cost.state_vector_bytes == 16 * 2 ** 5
    assert cost.peak_traced_bytes >= cost.pauli_operator_bytes


def test_the_cost_probe_quotes_a_price_and_never_a_verdict():
    """The bound is a policy on the referee, and the probe is not an escape hatch."""
    width = DENSE_SPEC_LIMIT + 1
    target = ghz_target(width)
    circuit = _non_clifford_ghz(width)
    cost = dense_tier_cost(circuit, target)
    assert cost.within_tolerance is True
    assert "verdict" not in cost.as_dict()
    # The referee still refuses to decide it, and the public oracle still refuses to
    # materialise it: two independent guards, neither of which the probe touches.
    report = adjudicate_specification(circuit, target)
    assert report.tier == "none" and report.verdict == "UNRESOLVED"
    with pytest.raises(SynthesisError, match="dense bound"):
        specification_state_vector(target)


def test_the_cost_probe_refuses_a_circuit_of_the_wrong_width():
    with pytest.raises(SynthesisError, match="cost probe needs"):
        dense_tier_cost(_non_clifford_ghz(4), ghz_target(5))


# ------------------------------------------------------------------- constructions
@pytest.mark.parametrize("family", sorted(set(FAMILIES) - REFUSAL_FAMILIES))
def test_every_family_reproduces_the_target_it_claims(family: str):
    target = _target_for(family)
    construction = build(family, target)
    assert construction.circuit.num_qubits == target.num_qubits
    assert len(construction.edges) == construction.circuit.resources()["two_qubit_gates"]
    report = adjudicate_specification(construction.circuit, target)
    assert report.verdict == "VALID", (family, report.failed())
    assert report.tier == "stabilizer"


def test_a_family_that_builds_another_target_is_refused_by_name():
    with pytest.raises(SynthesisError, match="constructs a 'ghz' state"):
        build("chain", ring_graph_target(5))
    with pytest.raises(SynthesisError, match="constructs a 'ring_graph' state"):
        build("ring", ghz_target(5))
    with pytest.raises(SynthesisError, match="unknown construction family"):
        build("teleportation", ghz_target(5))


def test_the_registered_near_misses_are_exactly_the_families_expected_to_fail():
    """The refusal registry is a list of deliberate wrong constructions."""
    assert REFUSAL_FAMILIES == {"linear_cluster", "cluster_drop_edge", "cluster_extra_edge",
                                "graph_swap_gate"}
    assert set(REFUSAL_FAMILIES) <= set(FAMILIES)
    outcome = synthesize(ring_graph_target(6))
    verdicts = {row["family"]: row["verdict"] for row in outcome.rows}
    assert verdicts == {"ring": "VALID", "linear_cluster": "INVALID"}
    assert outcome.promoted is not None
    assert outcome.promoted.endswith("-ring")


def test_the_ring_target_is_a_different_state_from_the_ghz_target():
    """`X Z Z` on every qubit is not `X X X` plus `ZZ`, and the referee can tell."""
    chain = build("chain", ghz_target(5))
    ring = build("ring", ring_graph_target(5))
    assert adjudicate_specification(chain.circuit, ring_graph_target(5)).verdict == "INVALID"
    assert adjudicate_specification(ring.circuit, ghz_target(5)).verdict == "INVALID"


def test_the_missing_closing_edge_is_refused_with_the_failing_generators_named():
    target = ring_graph_target(5)
    report = adjudicate_specification(build("linear_cluster", target).circuit, target)
    assert report.verdict == "INVALID"
    failed = next(check for check in report.checks if check.passed is False)
    assert failed.measured["failed_count"] == 2
    names = [row["pauli"] for row in failed.measured["failed"]]
    assert "+XZIIZ" in names and "+ZIIZX" in names


def _mutate(construction, *, operation: str):
    """The wrong constructions the record reports: each changes the prepared state."""
    ops = list(construction.circuit.ops)
    if operation == "drop_root_h":
        ops = [op for op in ops if getattr(op, "name", None) != "h"]
    elif operation == "reverse_middle_edge":
        index = len(ops) // 2
        target = ops[index]
        ops[index] = Gate(target.name, tuple(reversed(target.targets)))
    elif operation == "extra_h":
        ops.append(Gate("h", (construction.circuit.num_qubits - 1,)))
    elif operation == "cx_to_cz":
        index = next(i for i, op in enumerate(ops) if getattr(op, "name", None) == "cx")
        ops[index] = Gate("cz", ops[index].targets)
    else:  # pragma: no cover - the caller names the mutations
        raise AssertionError(operation)
    return Circuit(ops=tuple(ops), num_qubits=construction.circuit.num_qubits,
                   num_bits=0)


@pytest.mark.parametrize("operation", ("drop_root_h", "reverse_middle_edge", "extra_h",
                                       "cx_to_cz"))
def test_a_wrong_construction_is_refused_with_its_first_failing_stabilizer(operation: str):
    target = ghz_target(6)
    construction = build("chain", target)
    report = adjudicate_specification(_mutate(construction, operation=operation), target)
    assert report.verdict == "INVALID", operation
    failed = next(check for check in report.checks if check.passed is False)
    assert failed.measured["failed_count"] >= 1
    assert failed.measured["failed"][0]["pauli"]


# ------------------------------------------------------------------- the ranking
def test_the_ranking_and_frontier_are_measurements_not_preferences():
    target = ghz_target(64)
    outcome = synthesize(target)
    rows = {row["family"]: row for row in outcome.rows}
    assert all(row["verdict"] == "VALID" for row in outcome.rows)
    # Every family costs exactly one two-qubit gate per non-root qubit.
    assert {row["two_qubit_gates"] for row in outcome.rows} == {63}
    assert rows["chain"]["nearest_neighbour_only"] is True
    assert rows["star"]["nearest_neighbour_only"] is False
    assert rows["balanced_tree"]["depth"] < rows["chain"]["depth"]
    # The best-ranked construction is the shallowest, and the frontier keeps the
    # nearest-neighbour family even though it is deeper: they trade off.
    assert rows[outcome.ranked[0]]["depth"] == min(row["depth"] for row in outcome.rows)
    assert "chain" in outcome.pareto and outcome.ranked[0] in outcome.pareto
    assert rows["chain"]["max_edge_span"] == 1
    assert rows["balanced_tree"]["max_edge_span"] > 1
    # A star's control qubit serialises its gates: it is not shallower than a chain.
    assert rows["star"]["depth"] == rows["chain"]["depth"]


def test_synthesis_promotes_through_the_shared_admission_policy():
    outcome = synthesize(ghz_target(100))
    assert outcome.promoted is not None
    assert outcome.verdict is not None
    assert outcome.verdict.verdict == "VALID"
    assert outcome.verdict.claim_verdict == "MET"
    assert outcome.verdict.evidence == "SUFFICIENT"
    assert outcome.verdict.promotable is True
    assert outcome.verdict.tier == "stabilizer"
    payload = outcome.as_dict()
    assert payload["admission"]["promotion_decision"] == "PROMOTED"
    assert payload["admission"]["candidate_id"] == outcome.promoted
    assert payload["specification"]["name"] == "ghz"
    assert payload["promotion_refusal"] is None


def test_a_claim_the_construction_does_not_meet_is_unmet_but_still_valid():
    """The safeguards from the rewrites loop are the safeguards here."""
    outcome = synthesize(ghz_target(32), claims={"depth": 1})
    assert outcome.promoted is None
    assert outcome.verdict is not None
    assert outcome.verdict.verdict == "VALID"
    assert outcome.verdict.claim_verdict == "UNMET"
    assert outcome.verdict.promotable is False
    assert "does not improve its resource claim" in (outcome.refusal or "")
    assert "depth: claimed 1" in (outcome.refusal or "")


def test_the_claim_rule_is_the_same_rule_as_the_equivalence_referee_uses():
    circuit = build("chain", ghz_target(6)).circuit
    for claims in ({}, {"operations": 100}, {"operations": 1}, {"depth": 6, "operations": 6},
                   {"two_qubit_gates": 5}, {"measurements": 0}):
        mine = synthesis_claim_check(circuit, claims) if claims else None
        theirs = equivalence_claim_check(circuit, claims) if claims else None
        if mine is None:
            assert theirs is None
            continue
        assert mine.passed == theirs.passed
        assert mine.measured["unmet"] == theirs.measured["unmet"]


# ------------------------------------------------------- the dense tier boundary
def _non_clifford_ghz(num_qubits: int) -> Circuit:
    """A GHZ state prepared with `ry(pi/2)` instead of `h`: not a Clifford circuit."""
    ops: list = [Gate("ry", (0,), params=(math.pi / 2,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0)


def test_a_non_clifford_construction_is_decided_by_the_dense_tier():
    """Experiment C's boundary: verification already crosses it, the search does not."""
    circuit = _non_clifford_ghz(3)
    assert is_clifford_circuit(circuit) is False
    target = ghz_target(3)
    report = adjudicate_specification(circuit, target)
    assert report.tier == "dense"
    assert report.verdict == "VALID", report.failed()
    amplitude_check = next(check for check in report.checks
                           if "amplitudes" in check.name)
    assert amplitude_check.measured["distance"] <= 1e-12


def test_a_non_clifford_construction_that_is_wrong_is_refused_by_the_dense_tier():
    ops = [Gate("ry", (0,), params=(math.pi / 3,)),
           Gate("cx", (0, 1)), Gate("cx", (1, 2))]
    circuit = Circuit(ops=tuple(ops), num_qubits=3, num_bits=0)
    report = adjudicate_specification(circuit, ghz_target(3))
    assert report.tier == "dense" and report.verdict == "INVALID"


def test_above_the_dense_bound_a_non_clifford_construction_is_unresolved_not_wrong():
    circuit = _non_clifford_ghz(DENSE_SPEC_LIMIT + 2)
    report = adjudicate_specification(circuit, ghz_target(DENSE_SPEC_LIMIT + 2))
    assert report.tier == "none"
    assert report.verdict == "UNRESOLVED"
    assert report.failed() == ()
    check = report.checks[-1]
    assert check.passed is None and "above the dense specification bound" in check.measured["reason"]


def test_a_measuring_construction_has_no_state_claim():
    ops = [Gate("h", (0,)), Gate("cx", (0, 1)), Measure(0, 0)]
    circuit = Circuit(ops=tuple(ops), num_qubits=2, num_bits=1)
    report = adjudicate_specification(circuit, ghz_target(2))
    assert report.verdict == "UNRESOLVED"
    assert "prepares a distribution" in report.checks[-1].measured["reason"]


def test_the_wrong_qubit_count_is_a_failure_not_an_abstention():
    report = adjudicate_specification(_non_clifford_ghz(3), ghz_target(4))
    assert report.verdict == "INVALID"
    assert report.failed() == ("the construction acts on the specified number of qubits",)


# ------------------------------------------------------------------- no reference
def test_nothing_in_the_specification_path_carries_a_reference_circuit():
    candidate = SpecificationCandidate(candidate_id="c", circuit=_non_clifford_ghz(3),
                                       specification=ghz_target(3))
    payload = candidate.as_dict()
    assert "reference_sha256" not in payload
    assert "specification_sha256" in payload
    assert payload["specification"]["name"] == "ghz"
    # And the referee takes only a circuit and a specification.
    import inspect

    parameters = inspect.signature(adjudicate_specification).parameters
    assert list(parameters) == ["circuit", "target", "tier"]
    # `tier` chooses *which* referee answers the question; it is not another input to
    # it, and it cannot be passed positionally by accident.
    assert parameters["tier"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["tier"].default == "auto"
    assert not any("reference" in name or "holdout" in name for name in parameters)


def test_an_uncheckable_specification_candidate_is_refused_not_promoted():
    """The empty-check rule: `UNRESOLVED` cannot become `PROMOTED`."""
    circuit = _non_clifford_ghz(DENSE_SPEC_LIMIT + 1)
    candidate = SpecificationCandidate(candidate_id="uncheckable", circuit=circuit,
                                       specification=ghz_target(DENSE_SPEC_LIMIT + 1))
    verdict = adjudicate_specification_candidate(candidate)
    assert isinstance(verdict, CandidateVerdict)
    assert verdict.verdict == "UNRESOLVED"
    assert verdict.promotable is False
    with pytest.raises(PromotionRefused, match="not VALID"):
        promote(verdict)


def test_a_specification_candidate_satisfies_the_candidate_protocol():
    from apedp_dyn.vqc.discovery import Candidate

    candidate = SpecificationCandidate(candidate_id="c", circuit=_non_clifford_ghz(2),
                                       specification=ghz_target(2))
    assert isinstance(candidate, Candidate)


# --------------------------------------------------------------------- at scale
# ------------------------------------------------------------- referee agreement
@pytest.mark.parametrize("family", ("chain", "balanced_tree", "star"))
def test_both_referees_reach_the_same_verdict_at_every_width_both_can_decide(family):
    """Two representations, one question: group membership versus amplitudes.

    The stabilizer tier answers with exact integer arithmetic on ``n`` Pauli strings;
    the dense tier solves for ``2**n`` amplitudes and minimises a phase-free distance.
    They share the specification and nothing else, so agreement is evidence and a
    disagreement would mean the verdict is representation-dependent.
    """
    for num_qubits in (2, 3, 5, 8, DENSE_SPEC_LIMIT):
        target = ghz_target(num_qubits)
        agreement = referee_agreement(build(family, target).circuit, target)
        assert agreement.agree is True, (family, num_qubits, agreement.as_dict())
        assert agreement.stabilizer_verdict == agreement.dense_verdict == "VALID"
        assert agreement.stabilizer_checks_failed == agreement.dense_checks_failed == ()
        # Not "small": these are exact amplitudes, and the distance is rounding.
        assert agreement.dense_distance is not None
        assert agreement.dense_distance < 1e-12


def test_both_referees_refuse_the_same_wrong_construction():
    """Agreement on the accept side alone would be a referee that never says no."""
    target = ghz_target(5)
    wrong = Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)), Gate("cx", (1, 2)),
                         Gate("cx", (2, 3)), Gate("cx", (3, 4)),
                         Gate("h", (4,))), num_qubits=5, num_bits=0)
    agreement = referee_agreement(wrong, target)
    assert agreement.agree is True
    assert agreement.stabilizer_verdict == agreement.dense_verdict == "INVALID"
    assert len(agreement.stabilizer_checks_failed) >= 1
    assert len(agreement.dense_checks_failed) >= 1
    assert agreement.dense_distance > DENSE_TOLERANCE


def test_a_named_tier_that_cannot_decide_is_a_refusal_not_a_fallback():
    """A quiet fallback would let `referee_agreement` compare a referee with itself."""
    target = ghz_target(3)
    non_clifford = Circuit(ops=(Gate("ry", (0,), params=(math.pi / 2,)),
                               Gate("cx", (0, 1)), Gate("cx", (1, 2))),
                          num_qubits=3, num_bits=0)
    with pytest.raises(SynthesisError, match="stabilizer tier cannot decide"):
        adjudicate_specification(non_clifford, target, tier="stabilizer")
    with pytest.raises(SynthesisError, match="tier must be"):
        adjudicate_specification(non_clifford, target, tier="fastest")
    # A measuring circuit has no state for either tier to compare.
    measuring = Circuit(ops=(Gate("h", (0,)), Measure(0, 0)), num_qubits=1, num_bits=1)
    plus = TargetSpecification(name="plus", num_qubits=1, stabilizers=(((1,), 0),))
    with pytest.raises(SynthesisError, match="measures"):
        adjudicate_specification(measuring, plus, tier="dense")
    # The dense tier refuses a width it cannot hold rather than materialising it.
    wide = Circuit(ops=(Gate("h", (0,)),
                        *[Gate("cx", (q, q + 1)) for q in range(DENSE_SPEC_LIMIT)]),
                   num_qubits=DENSE_SPEC_LIMIT + 1, num_bits=0)
    with pytest.raises(SynthesisError, match="bounded at"):
        adjudicate_specification(wide, ghz_target(DENSE_SPEC_LIMIT + 1), tier="dense")
    # And `referee_agreement` cannot be asked for an agreement it cannot test.
    with pytest.raises(SynthesisError):
        referee_agreement(non_clifford, target)


def test_the_dense_referee_is_normalised_and_blind_to_one_global_phase():
    """`X Y Z = iI`: multiplying the whole state by `i` is not a different state."""
    target = ghz_target(4)
    plain = build("chain", target).circuit
    rotated = Circuit(ops=(*plain.ops, Gate("x", (0,)), Gate("y", (0,)), Gate("z", (0,))),
                      num_qubits=4, num_bits=0)
    agreement = referee_agreement(rotated, target)
    assert agreement.agree is True and agreement.dense_verdict == "VALID"
    # The requirement string carries the tolerance, so the report says what it accepted at.
    report = adjudicate_specification(rotated, target, tier="dense")
    check = report.checks[-1]
    assert f"{DENSE_TOLERANCE:g}" in check.requirement
    assert "global phase" in check.name
    assert check.measured["relative_phase_sensitive"] is True


def test_a_thousand_qubit_specification_is_decided_exactly():
    target = ghz_target(1000)
    outcome = synthesize(target, families=("chain", "balanced_tree"))
    assert outcome.promoted is not None
    rows = {row["family"]: row for row in outcome.rows}
    assert rows["chain"]["operations"] == 1000
    assert rows["balanced_tree"]["depth"] < rows["chain"]["depth"]
    assert all(row["verdict"] == "VALID" for row in outcome.rows)
    # The exact stabilizer check ran, generator by generator, with no approximation.
    report = adjudicate_specification(build("chain", target).circuit, target)
    assert report.verdict == "VALID"
    assert report.checks[1].measured["stabilizers_checked"] == 1000
    state = StabilizerState(1000)
    for op in build("chain", target).circuit.ops:
        state.apply_gate(op.name, op.targets, op.params)
    assert state.expectation(tuple([1] * 1000)) == 1.0


# ------------------------------------------------------- experiment E: graph states
HUB_EDGES = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    *((qubit, qubit + 1) for qubit in range(8, 19)),
    (7, 8), (2, 11), (4, 15), (6, 19), (9, 14), (1, 12),
)


def test_a_two_dimensional_cluster_is_the_grid_graph_state():
    target = cluster_state_target(3, 3)
    assert target.name == "cluster_2d" and target.num_qubits == 9
    assert graph_of_state(target) == tuple(sorted(set(lattice_edges(3, 3))))
    assert len(graph_of_state(target)) == 12
    # Every stabilizer is X on exactly one qubit and Z on its neighbours.
    for labels, phase in target.stabilizers:
        assert phase == 0
        assert sum(1 for label in labels if label == 1) == 1     # X
        assert all(label in (0, 1, 3) for label in labels)       # I, X, Z only
    # A 3x3 lattice is invariant under reversing its wires, so it cannot expose a
    # basis-order mistake -- which is exactly why Experiment E uses irregular graphs.
    assert entangling_depth_lower_bound(target)["cz_layers_floor"] == 4


def test_a_graph_that_is_not_a_state_preparation_is_refused():
    with pytest.raises(SynthesisError, match="self-loop"):
        irregular_graph_target(4, [(0, 0), (0, 1), (1, 2), (2, 3)])
    with pytest.raises(SynthesisError, match="twice"):
        irregular_graph_target(4, [(0, 1), (0, 1), (1, 2), (2, 3)])
    with pytest.raises(SynthesisError, match="out of range"):
        irregular_graph_target(4, [(0, 4)])
    with pytest.raises(SynthesisError, match="disconnected"):
        irregular_graph_target(4, [(0, 1)])
    with pytest.raises(SynthesisError, match="positive dimensions"):
        lattice_edges(0, 3)


def test_the_graph_parser_refuses_a_specification_that_is_not_a_graph_state():
    with pytest.raises(SynthesisError, match="X positions"):
        graph_of_state(ghz_target(4))
    # `Y_0` on qubit 0 is a legitimate +1 stabilizer -- of |+i>|0>, not of a graph state.
    with_y = TargetSpecification(name="with_y", num_qubits=2,
                                 stabilizers=(((2, 0), 0), ((0, 3), 0)))
    with pytest.raises(SynthesisError, match="not of the form"):
        graph_of_state(with_y)
    # The parser reads the wiring from the stabilizers, so it must round-trip.
    target = irregular_graph_target(6, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
    assert graph_of_state(target) == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))


def test_a_graph_family_refuses_the_target_it_was_not_declared_for():
    with pytest.raises(SynthesisError, match=r"constructs a 'ghz' state"):
        build("chain", cluster_state_target(2, 3))
    with pytest.raises(SynthesisError, match="constructs a 'cluster_2d' state"):
        build("graph_layers", ring_graph_target(6))


def test_every_valid_graph_family_prepares_the_cluster_state():
    target = cluster_state_target(3, 4)
    families = [name for name in default_families(target) if name not in REFUSAL_FAMILIES]
    assert len(families) == 5, families
    for family in families:
        construction = build(family, target)
        report = adjudicate_specification(construction.circuit, target)
        assert report.verdict == "VALID", (family, report.failed())
        assert report.tier == "stabilizer"


def test_the_schedule_is_the_only_free_variable_and_it_changes_the_depth():
    """The specification fixes the wiring, so all families cost the same and differ in depth."""
    target = cluster_state_target(6, 6)
    rows = []
    for family in ("graph_raster", "graph_column_major", "graph_tree_first",
                   "graph_layers", "graph_random_order"):
        construction = build(family, target)
        rows.append((family, construction.operations, construction.two_qubit_gates,
                     construction.max_edge_span, construction.depth))
    assert len({row[1] for row in rows}) == 1, "the operation count is the specification's"
    assert len({row[2] for row in rows}) == 1
    assert len({row[3] for row in rows}) == 1, "the connectivity is the specification's"
    assert len({row[4] for row in rows}) > 1, "the depth is the schedule's"
    depths = {row[0]: row[4] for row in rows}
    assert depths["graph_layers"] < depths["graph_raster"]
    assert depths["graph_layers"] == 1 + entangling_depth_lower_bound(target)["cz_layers_floor"]


def test_the_peeled_schedule_attains_the_degree_floor_at_every_size():
    for rows, cols in ((3, 3), (4, 4), (5, 5), (8, 8)):
        target = cluster_state_target(rows, cols)
        floor = entangling_depth_lower_bound(target)["cz_layers_floor"]
        construction = build("graph_layers", target)
        assert construction.depth == 1 + floor, (rows, cols, construction.depth, floor)
        assert adjudicate_specification(construction.circuit, target).verdict == "VALID"


def test_the_layer_schedule_is_proved_minimal_on_a_class_two_graph():
    """K5 has Delta = 4 but chi' = 5: the schedule must prove the fifth layer."""
    complete = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
                (2, 3), (2, 4), (3, 4))
    target = irregular_graph_target(5, complete)
    construction = build("graph_layers", target)
    assert construction.depth == 1 + 5
    assert "proven minimal" in construction.notes
    assert "greedy peel used 7" in construction.notes
    assert adjudicate_specification(construction.circuit, target).verdict == "VALID"


def test_an_irregular_graph_state_is_built_and_attains_its_own_floor():
    target = irregular_graph_target(20, HUB_EDGES)
    floor = entangling_depth_lower_bound(target)
    assert floor["edges"] == len(HUB_EDGES)
    assert floor["cz_layers_floor"] == 7
    construction = build("graph_layers", target)
    assert construction.depth == 1 + floor["cz_layers_floor"]
    assert adjudicate_specification(construction.circuit, target).verdict == "VALID"
    seeded = seeded_graph_target(24, seed=7)
    assert graph_of_state(seeded) == graph_of_state(seeded_graph_target(24, seed=7))
    assert len(graph_of_state(seeded)) > 24


def test_the_wrong_graph_constructions_are_refused_with_the_failing_edges_named():
    target = cluster_state_target(3, 3)
    dropped = adjudicate_specification(build("cluster_drop_edge", target).circuit, target)
    extra = adjudicate_specification(build("cluster_extra_edge", target).circuit, target)
    swapped = adjudicate_specification(build("graph_swap_gate", target).circuit, target)
    for report in (dropped, extra, swapped):
        assert report.verdict == "INVALID"
        failed = next(check for check in report.checks if check.passed is False)
        assert failed.measured["failed_count"] >= 2
        assert failed.measured["failed"][0]["pauli"]
    def failure_count(report) -> int:
        check = next(check for check in report.checks
                     if "eigenstate" in check.name)
        return check.measured["failed_count"]

    # A missing or an extra edge costs the stabilizers of its two endpoints -- and only
    # those, which is what makes the refusal diagnostic rather than just a negative.
    assert failure_count(dropped) == 2
    assert failure_count(extra) == 2
    assert failure_count(swapped) >= 2


def test_the_two_basis_orders_are_pinned_on_an_asymmetric_state():
    """The dense tier compared a state with its bit reversal, and Experiment E found it.

    A three-qubit path graph is the smallest specification that is *not* invariant
    under reversing its wires, so it is the smallest state that can catch the mistake:
    the re-indexed amplitudes are the specification, the un-re-indexed ones are not.
    """
    target = irregular_graph_target(3, ((0, 1), (0, 2)))
    circuit = build("graph_layers", target).circuit
    expected = specification_state_vector(target)
    prepared = _prepare_dense(circuit)
    good, _phase = _phase_free_distance(prepared, expected)
    naive, _phase = _phase_free_distance(_specification_order(prepared, 3), expected)
    assert good < 1e-12
    assert naive > 1e-3, "without the re-index the comparison must be visibly wrong"
    agreement = referee_agreement(circuit, target)
    assert agreement.agree is True
    assert agreement.stabilizer_verdict == agreement.dense_verdict == "VALID"
    assert agreement.dense_distance < 1e-12


def test_both_referees_agree_on_an_asymmetric_graph_state():
    """The same statement at a width where a reversal is no longer an automorphism."""
    target = seeded_graph_target(9, seed=3)
    circuit = build("graph_layers", target).circuit
    agreement = referee_agreement(circuit, target)
    assert agreement.agree is True
    assert agreement.stabilizer_verdict == agreement.dense_verdict == "VALID"
    assert agreement.stabilizer_checks_failed == agreement.dense_checks_failed == ()
