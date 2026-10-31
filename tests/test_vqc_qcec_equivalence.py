"""MQT QCEC on the equivalence task, over a corpus built so the two disagree.

The rules this file pins are the ones the comparison harness states: a tool that did
not run is not a tool that lost, and an *attributable* disagreement is worth more than
an agreement. So the corpus test does not assert the two referees agree. It asserts
that the construction which built each pair fixes the answer, that our referee answers
what the construction requires on every scope it offers, and that each divergence in
the record is named -- a threshold, a scope, or an abstention on one side:

* **the construction is the arbiter.** Every pair carries an algebraic identity (``X Y
  Z = i I``, ``RZ(2*pi) = -I``, ``SWAP = CX CX CX``, ``U_c = RX(theta) U_r`` with a
  non-scalar rotation), so an expectation of `VALID` is the construction's and not the
  tool's;
* **a threshold is a policy, not a truth.** The ladder is one angle from ``1e-13`` to
  ``1e-3``, every point of it exactly inequivalent, and the record carries both
  distances and both thresholds rather than one verdict as if it were the answer;
* **a scope difference is not a disagreement.** The pair that prepares one state from
  two unitaries is where a naive comparison would record a contradiction, so it is
  pinned twice: our state-scope verdict is `VALID`, our unitary-scope verdict is
  `INVALID`, and the other tool's answer agrees with the second;
* **abstentions are recorded on the side that abstains.** Above the dense bound a
  non-Clifford pair is refused here and decided by QCEC, and neither fact is scored.

The QCEC-dependent tests are skipped when it is not installed. That is not a gap: the
corpus, the classifier, the summary and the record are pinned without it, and the
record's own rows are checked against the corpus expectations either way.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.vqc.circuit import Circuit, Gate, Measure
from apedp_dyn.vqc.equivalence import (
    MAX_UNITARY_QUBITS,
    NotAUnitaryCircuitError,
    adjudicate_clifford_equivalence,
    unitary,
)
from apedp_dyn.vqc.equivalence_corpus import (
    EQUIVALENCE_TOLERANCE,
    SCOPES,
    EquivalencePair,
    adjudicate_pair,
    classify,
    equivalence_pairs,
    summarise,
)
from apedp_dyn.vqc.external import (
    TOOLS,
    ForeignEquivalence,
    ToolUnavailable,
    probe,
    qcec_equivalence,
    qiskit_circuit,
)

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_external_comparison.json"
MODULE = ROOT / "apedp_dyn" / "vqc" / "equivalence_corpus.py"
#: The criteria QCEC's manager can return, as the record records them.
CRITERIA = {"equivalent", "equivalent_up_to_global_phase", "not_equivalent",
            "no_information"}


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the corpus
def test_every_pair_carries_the_construction_that_fixes_its_relation():
    pairs = equivalence_pairs()
    assert len(pairs) >= 12
    assert {pair.relation for pair in pairs} == {
        "equal", "equal_up_to_global_phase", "same_prepared_state", "unequal"}
    for pair in pairs:
        assert pair.construction, pair.label
        assert pair.question in SCOPES, pair.label
        assert pair.expected, pair.label
        assert set(pair.expected) <= set(SCOPES), pair.label
        assert pair.reference.num_qubits == pair.candidate.num_qubits, pair.label


def test_our_referee_answers_what_every_construction_requires():
    """The corpus is a test of this package first: the other tool is the comparison."""
    failures = []
    for pair in equivalence_pairs():
        verdicts = adjudicate_pair(pair)
        for scope, expected in pair.expected.items():
            if verdicts[scope].verdict != expected:
                failures.append((pair.label, scope, verdicts[scope].verdict, expected))
    assert not failures, failures


def test_all_three_scopes_are_answered_or_refused_with_a_reason():
    for pair in equivalence_pairs():
        verdicts = adjudicate_pair(pair)
        assert set(verdicts) == set(SCOPES), pair.label
        for scope, verdict in verdicts.items():
            if verdict.verdict == "UNRESOLVED":
                assert verdict.reason, (pair.label, scope)
            assert verdict.decides, (pair.label, scope)


def test_above_the_dense_bound_a_non_clifford_pair_is_refused_by_name():
    pair = next(pair for pair in equivalence_pairs()
                if pair.label == "wide-non-clifford-differing")
    assert pair.qubits > MAX_UNITARY_QUBITS
    standalone = adjudicate_pair(pair)["standalone"]
    assert standalone.verdict == "UNRESOLVED"
    # The reason names the gate, so a reader sees which tier ran out and why.
    assert "'t' is not Clifford" in standalone.reason
    # And the refusal is a refusal: the stabilizer report itself would say INVALID.
    report = adjudicate_clifford_equivalence(pair.reference, pair.candidate)
    assert report.verdict == "INVALID"
    assert any(check.name.startswith("reference is a Clifford")
               for check in report.checks if check.passed is False)


def test_a_clifford_pair_above_the_dense_bound_is_still_decided():
    pair = next(pair for pair in equivalence_pairs()
                if pair.label == "wide-clifford-rewrite")
    assert pair.qubits > MAX_UNITARY_QUBITS
    verdict = adjudicate_pair(pair)["standalone"]
    assert verdict.verdict == "VALID"
    assert verdict.tier == "stabilizer"
    assert verdict.distance is None, "the conjugation action has no numeric distance"


def test_the_tolerance_ladder_is_one_angle_and_both_edges_are_inside_it():
    ladder = [pair for pair in equivalence_pairs() if pair.family == "tolerance"]
    assert len(ladder) >= 5
    distances = [adjudicate_pair(pair)["standalone"].distance for pair in ladder]
    assert all(distance is not None and distance > 0 for distance in distances)
    assert distances == sorted(distances), "the ladder is monotone in the angle"
    # The ladder straddles this package's own tolerance, which is the point: at least
    # one point is inside it and at least one outside, and every point is unequal.
    verdicts = [pair.expected["standalone"] for pair in ladder]
    assert "VALID" in verdicts and "INVALID" in verdicts
    assert all(pair.relation == "unequal" for pair in ladder)
    inside = [d for d, v in zip(distances, verdicts, strict=True) if v == "VALID"]
    assert inside and max(inside) <= EQUIVALENCE_TOLERANCE
    outside = [d for d, v in zip(distances, verdicts, strict=True) if v == "INVALID"]
    assert outside and min(outside) > EQUIVALENCE_TOLERANCE


def test_the_state_pair_is_equal_as_a_state_and_different_as_an_operation():
    """The pair a naive comparison would call a contradiction, decided twice."""
    pair = next(pair for pair in equivalence_pairs() if pair.family == "state")
    assert pair.question == "state"
    verdicts = adjudicate_pair(pair)
    assert verdicts["state"].verdict == "VALID"
    assert verdicts["standalone"].verdict == "INVALID"
    assert verdicts["controlled"].verdict == "INVALID"
    assert verdicts["standalone"].distance > 0


def test_a_phase_difference_invisible_to_measurement_is_still_a_difference():
    """The diagonal pair: identical on every Z basis outcome, different unitaries.

    Both circuits are products of diagonal gates, so every computational-basis
    probability agrees; only a unitary referee separates them, and both do.
    """
    pair = next(pair for pair in equivalence_pairs() if pair.label == "relative-diagonal-cz")
    verdicts = adjudicate_pair(pair)
    assert verdicts["standalone"].verdict == "INVALID"
    assert verdicts["state"].verdict == "INVALID"
    # The claim the construction makes, measured: the prepared amplitudes have the
    # same magnitudes -- every Z-basis probability agrees -- and different phases. The
    # two states are therefore orthogonal, so a state-scope VALID here would be a
    # false VALID; test_vqc_stabilizer.py guards that exhaustively.
    first = unitary(pair.reference)[:, 0]
    second = unitary(pair.candidate)[:, 0]
    assert np.allclose(np.abs(first), np.abs(second), atol=1e-12)
    assert not np.allclose(first, second, atol=1e-12)
    assert abs(np.vdot(first, second)) == pytest.approx(0.0, abs=1e-12)


def test_the_global_phase_pairs_are_free_standalone_and_not_under_control():
    pairs = [pair for pair in equivalence_pairs() if pair.family == "global_phase"]
    assert len(pairs) == 2
    for pair in pairs:
        verdicts = adjudicate_pair(pair)
        assert pair.relation == "equal_up_to_global_phase"
        assert verdicts["standalone"].verdict == "VALID"
        assert verdicts["standalone"].distance <= EQUIVALENCE_TOLERANCE
        assert verdicts["controlled"].verdict == "INVALID"
        assert verdicts["controlled"].scope == "controlled"


# ---------------------------------------------------------------- the classifier
def _pair(label: str) -> EquivalencePair:
    return next(pair for pair in equivalence_pairs() if pair.label == label)


def _classified(label: str, foreign: str | None, criteria: tuple[str, ...] = ()):
    pair = _pair(label)
    return classify(pair, adjudicate_pair(pair), foreign_verdict=foreign,
                    foreign_criteria=criteria)


def test_agreement_is_agreement_on_both_answers():
    assert _classified("rewrite-cancelled-pair", "equivalent")["outcome"] == "AGREE"
    assert _classified("relative-s-versus-sdg", "not_equivalent")["outcome"] == "AGREE"
    phase = _classified("phase-xyz-is-i", "equivalent_up_to_global_phase")
    assert phase["outcome"] == "AGREE"
    assert "global phase" in phase["explanation"]


def test_a_threshold_difference_is_not_a_contradiction_and_not_an_agreement():
    outcome = _classified("tolerance-1e-07", "equivalent")
    assert outcome["outcome"] == "DIFFER_BY_THRESHOLD"
    assert outcome["axis"] == "tolerance"
    assert "exactly inequivalent" in outcome["explanation"]


def test_a_disagreement_that_no_threshold_explains_is_called_one():
    """The classifier is not allowed to explain every difference away."""
    outcome = _classified("rewrite-cancelled-pair", "not_equivalent")
    assert outcome["outcome"] == "CONTRADICTION"
    assert outcome["axis"] == "unexplained"


def test_the_state_question_is_a_scope_difference_not_a_disagreement():
    outcome = _classified("state-ghz-stabiliser", "not_equivalent")
    assert outcome["outcome"] == "SCOPE"
    assert outcome["compared"] is False
    assert "unitary" in outcome["explanation"]


def test_an_abstention_on_either_side_is_recorded_on_that_side():
    ours = _classified("wide-non-clifford-differing", "not_equivalent")
    assert ours["outcome"] == "WE_ABSTAIN"
    assert "cannot reach" in ours["explanation"]
    theirs = _classified("tolerance-1e-13", "no_information")
    assert theirs["outcome"] == "QCEC_ABSTAINS"
    assert theirs["compared"] is False
    both = _classified("wide-non-clifford-cancelled", "no_information")
    assert both["outcome"] == "BOTH_ABSTAIN"
    assert both["axis"] == "abstention"


def test_an_unstable_foreign_answer_is_reported_as_unstable():
    outcome = _classified("tolerance-1e-06", "unstable",
                          ("equivalent", "no_information"))
    assert outcome["outcome"] == "UNSTABLE"
    assert outcome["compared"] is False
    assert "no_information" in outcome["explanation"]


def test_a_tool_that_never_ran_is_an_abstention_with_no_verdict_attached():
    outcome = _classified("rewrite-cancelled-pair", None)
    assert outcome["outcome"] == "UNAVAILABLE"
    assert outcome["axis"] == "environment"
    assert outcome["compared"] is False


# ----------------------------------------------------------------- the summary
def test_the_summary_recomputes_from_the_rows_and_notices_a_bad_expectation():
    rows = []
    for pair in equivalence_pairs():
        verdicts = adjudicate_pair(pair)
        rows.append({**pair.as_dict(),
                     "ours": {scope: verdict.as_dict()
                              for scope, verdict in verdicts.items()},
                     "qcec": {"status": "unavailable", "verdict": None},
                     "agreement": classify(pair, verdicts, foreign_verdict=None)})
    summary = summarise(rows)
    assert summary["expected_met"] == []
    assert summary["pairs"] == len(rows)
    assert summary["contradictions"] == []
    assert summary["our_decided"] + summary["our_abstained"] == len(rows)
    # A tampered verdict is caught by the same function, rather than averaged away.
    rows[0]["ours"][rows[0]["question"]]["verdict"] = "INVALID"
    assert summarise(rows)["expected_met"], "a wrong verdict must be reported"


# ------------------------------------------------------------------ the adapter
def test_the_qiskit_gate_table_is_total_over_the_ir():
    from apedp_dyn.vqc.external import _QISKIT_GATES
    from apedp_dyn.vqc.gates import GATE_SPECS

    assert set(GATE_SPECS) == set(_QISKIT_GATES), (
        "a gate of the IR with no mapping would reach the other tool as a silent "
        "omission")


def test_a_circuit_with_every_gate_is_handed_over_gate_for_gate():
    pytest.importorskip("qiskit")
    ops = (Gate("h", (0,)), Gate("x", (1,)), Gate("y", (2,)), Gate("z", (0,)),
           Gate("s", (1,)), Gate("sdg", (2,)), Gate("t", (0,)), Gate("tdg", (1,)),
           Gate("rx", (2,), (0.3,)), Gate("ry", (0,), (0.2,)),
           Gate("rz", (1,), (0.1,)), Gate("phase", (2,), (0.7,)),
           Gate("cx", (0, 1)), Gate("cz", (1, 2)), Gate("swap", (0, 2)),
           Gate("ccx", (0, 1, 2)), Gate("cswap", (2, 0, 1)))
    circuit = Circuit(num_qubits=3, ops=ops)
    handed = qiskit_circuit(circuit)
    assert handed.num_qubits == circuit.num_qubits
    assert sum(handed.count_ops().values()) == len(circuit.ops)


def test_a_channel_cannot_be_handed_to_a_unitary_checker():
    measured = Circuit(num_qubits=2, ops=(Gate("h", (0,)), Measure(0, 0)),
                       num_bits=1)
    with pytest.raises(NotAUnitaryCircuitError):
        qiskit_circuit(measured)


def test_a_missing_qcec_is_a_refusal_with_a_reason(monkeypatch):
    from apedp_dyn.vqc import external

    def unavailable():
        raise ToolUnavailable("mqt.qcec is not importable (ImportError: test)")

    monkeypatch.setattr(external, "_import_qcec", unavailable)
    with pytest.raises(ToolUnavailable, match="not importable"):
        qcec_equivalence(Circuit(num_qubits=1, ops=()), Circuit(num_qubits=1, ops=()))


def test_an_unstable_verdict_is_not_reported_as_a_verdict():
    """The aggregation rule, pinned without the package: two answers are not one."""
    stable = ForeignEquivalence(tool="mqt.qcec", version="9", criteria=("equivalent",),
                                considered=(True,), seconds=(0.1,), checkers=())
    assert stable.verdict == "equivalent"
    assert stable.decided is True
    mixed = ForeignEquivalence(tool="mqt.qcec", version="9",
                               criteria=("equivalent", "no_information"),
                               considered=(True, False), seconds=(0.1, 0.2), checkers=())
    assert mixed.verdict == "unstable"
    assert mixed.decided is False
    assert mixed.as_dict()["repeats"] == 2
    assert mixed.as_dict()["criteria"] == ["equivalent", "no_information"]


def test_qcec_is_probed_as_wired_and_as_a_comparison():
    status = probe("mqt.qcec")
    assert TOOLS["mqt.qcec"]["wired"] is True
    if status.installed:
        assert status.status == "compared"
        assert status.comparative is True
    else:
        assert status.reason


# -------------------------------------------------------------- the qcec answers
def test_qcec_decides_the_pairs_the_way_the_corpus_says_it_can():
    """Its three vocabulary distinctions, pinned: they are why scope is recorded."""
    pytest.importorskip("mqt.qcec")
    pytest.importorskip("qiskit")
    assert qcec_equivalence(_pair("rewrite-cancelled-pair").reference,
                            _pair("rewrite-cancelled-pair").candidate).verdict == \
        "equivalent"
    phase = qcec_equivalence(_pair("phase-rz-2pi-is-minus-one").reference,
                             _pair("phase-rz-2pi-is-minus-one").candidate)
    assert phase.verdict == "equivalent_up_to_global_phase"
    assert phase.decided is True
    relative = qcec_equivalence(_pair("relative-diagonal-cz").reference,
                                _pair("relative-diagonal-cz").candidate)
    assert relative.verdict == "not_equivalent"


def test_qcec_gets_the_same_pairs_this_package_refuses_and_still_decides_them():
    pytest.importorskip("mqt.qcec")
    pytest.importorskip("qiskit")
    for label in ("wide-non-clifford-cancelled", "wide-non-clifford-differing"):
        pair = _pair(label)
        assert adjudicate_pair(pair)["standalone"].verdict == "UNRESOLVED"
        verdict = qcec_equivalence(pair.reference, pair.candidate)
        assert verdict.criteria[0] in CRITERIA
        assert verdict.seconds[0] > 0


def test_repeats_are_repeats_of_the_identical_call():
    pytest.importorskip("mqt.qcec")
    pytest.importorskip("qiskit")
    pair = _pair("rewrite-cancelled-pair")
    result = qcec_equivalence(pair.reference, pair.candidate, repeats=4, seed=1)
    assert len(result.criteria) == 4
    assert len(result.seconds) == 4
    # A pair nowhere near a threshold is stable, which is what makes the unstable
    # rows in the record informative rather than noise.
    assert result.verdict == "equivalent"
    assert result.as_dict()["repeats"] == 4


def test_a_configuration_that_is_not_the_tools_own_is_still_its_answer():
    pytest.importorskip("mqt.qcec")
    pytest.importorskip("qiskit")
    pair = _pair("rewrite-cancelled-pair")
    result = qcec_equivalence(pair.reference, pair.candidate, timeout=30.0,
                              numerical_tolerance=1e-12)
    assert result.timeout == 30.0
    assert result.numerical_tolerance == 1e-12
    assert result.verdict == "equivalent"


# -------------------------------------------------------------------- the record
def test_the_record_carries_the_equivalence_task_and_every_check_passed(record):
    assert record["schema"] == "apedp.vqc.external_comparison.v8"
    assert all(record["checks"].values())
    block = record["equivalence"]
    assert block["tolerance"] == EQUIVALENCE_TOLERANCE
    assert block["foreign"]["tool"] == "mqt.qcec"
    assert block["foreign"]["corpus_sha256"] == hashlib.sha256(
        MODULE.read_bytes()).hexdigest()
    assert block["design"]["pairs"] == len(block["rows"]) == len(equivalence_pairs())
    assert block["design"]["repeats"] >= 5


def test_every_recorded_verdict_is_the_one_the_construction_requires(record):
    for row in record["equivalence"]["rows"]:
        assert set(row["ours"]) == set(SCOPES), row["label"]
        assert row["construction"] and row["relation"] and row["expected"]
        for scope, expected in row["expected"].items():
            assert row["ours"][scope]["verdict"] == expected, (row["label"], scope)
        if row["ours"][row["question"]]["verdict"] == "UNRESOLVED":
            assert row["ours"][row["question"]]["reason"], row["label"]


def test_the_recorded_summary_recomputes_and_leaves_nothing_unexplained(record):
    block = record["equivalence"]
    assert summarise(block["rows"]) == block["summary"]
    assert block["summary"]["contradictions"] == []
    assert block["summary"]["expected_met"] == []
    assert block["summary"]["outcomes"].get("AGREE")
    assert len(block["summary"]["outcomes"]) >= 3, (
        "a corpus that only agrees is not the comparison this task asked for")


def test_every_repeated_row_carries_the_criteria_it_was_asked_for(record):
    block = record["equivalence"]
    repeats = block["design"]["repeats"]
    for row in block["rows"]:
        if row["family"] != block["design"]["repeated_family"]:
            continue
        assert row["qcec"]["repeats"] == repeats, row["label"]
        assert set(row["qcec"]["criteria"]) <= CRITERIA, row["label"]
        if row["agreement"]["outcome"] == "UNSTABLE":
            assert len(set(row["qcec"]["criteria"])) > 1, row["label"]


def test_a_compared_record_row_carries_the_tool_version_and_no_loss_is_counted(record):
    block = record["equivalence"]
    compared = [row for row in block["rows"] if row["qcec"]["status"] == "compared"]
    assert compared, "the record must hold the comparison it was built for"
    for row in compared:
        assert row["qcec"]["version"] == block["foreign"]["version"]
        assert row["qcec"]["verdict"] in CRITERIA | {"unstable"}
    # An abstention on either side is not a loss: it appears only as an outcome.
    for row in block["rows"]:
        if row["qcec"]["status"] != "compared":
            assert row["qcec"]["reason"]
            assert row["agreement"]["outcome"] == "UNAVAILABLE"
        if row["ours"][row["question"]]["verdict"] == "UNRESOLVED":
            assert row["agreement"]["outcome"] in ("WE_ABSTAIN", "BOTH_ABSTAIN")


def test_the_record_says_which_question_each_tool_answered(record):
    block = record["equivalence"]
    verdict = record["verdict"]["equivalence"]
    assert "threshold" in verdict
    assert "No divergence is left unexplained" in verdict
    assert "scope" in TOOLS["mqt.qcec"]["note"] or "controlled" in \
        TOOLS["mqt.qcec"]["note"]
    for row in block["rows"]:
        assert row["ours"][row["question"]]["decides"], row["label"]
