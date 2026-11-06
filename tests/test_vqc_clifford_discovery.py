"""The discovery loop: propose, certify, rank, promote — and refuse.

Two things are pinned. The **rules** are checked by feeding every rewrite of random
Clifford circuits to the referee: a rule that is not semantics-preserving fails the
certification instead of being trusted, so this is a property test of the rule set,
not of one example. The **workflow** is checked against the recorded 100-qubit run
and re-run cheaply at 8 qubits, including the candidate that must be refused.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from apedp_dyn.vqc import Circuit, Gate
from apedp_dyn.vqc.clifford_search import (
    SELF_INVERSE,
    propose_rewrites,
    resource_score,
    search,
)
from apedp_dyn.vqc.equivalence import adjudicate_clifford_equivalence
from apedp_dyn.vqc.stabilizer import CLIFFORD_GATES

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_clifford_discovery.json"


def junked_chain(num_qubits: int) -> Circuit:
    """The same workload the record uses: a GHZ chain with cancelling work spliced in."""
    ops: list = [Gate("h", (0,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    for qubit in range(0, num_qubits, 5):
        ops.extend([Gate("z", (qubit,)), Gate("z", (qubit,))])
    for qubit in range(1, num_qubits, 7):
        ops.extend([Gate("s", (qubit,)) for _ in range(4)])
    ops.append(Gate("h", (num_qubits - 1,)))
    for qubit in range(0, num_qubits, 11):
        ops.extend([Gate("z", (qubit,)), Gate("z", (qubit,))])
    ops.append(Gate("h", (num_qubits - 1,)))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def random_clifford(num_qubits: int, *, seed: int, depth: int = 14) -> Circuit:
    rng = random.Random(seed)
    ops = [Gate("h", (qubit,)) for qubit in range(num_qubits)]
    for _ in range(depth):
        name = rng.choice(sorted(CLIFFORD_GATES))
        if name in ("cx", "cz", "swap"):
            first, second = rng.sample(range(num_qubits), 2)
            ops.append(Gate(name, (first, second)))
        else:
            ops.append(Gate(name, (rng.randrange(num_qubits),)))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


# ------------------------------------------------------------------ the rule set
@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4))
def test_every_proposed_rewrite_is_the_same_unitary(seed):
    """A rewrite rule that changes the unitary must fail here, not be trusted."""
    circuit = random_clifford(4, seed=seed)
    rewrites = list(propose_rewrites(circuit))
    for rule, candidate in rewrites:
        report = adjudicate_clifford_equivalence(circuit, candidate)
        assert report.verdict == "VALID", (rule, report.mismatches[:1])
        assert resource_score(candidate) <= resource_score(circuit), rule


def test_the_rewrites_remove_work_and_never_add_it():
    circuit = junked_chain(12)
    before = resource_score(circuit)
    for _rule, candidate in propose_rewrites(circuit):
        assert resource_score(candidate) < before


@pytest.mark.parametrize(
    ("gates", "expected_gate"),
    (
        ((Gate("cx", (0, 1)), Gate("cx", (0, 2)), Gate("cx", (0, 1))),
         Gate("cx", (0, 2))),
        ((Gate("cx", (0, 2)), Gate("cx", (1, 2)), Gate("cx", (0, 2))),
         Gate("cx", (1, 2))),
        ((Gate("cz", (0, 1)), Gate("cz", (1, 2)), Gate("cz", (0, 1))),
         Gate("cz", (1, 2))),
    ),
)
def test_commuting_entanglers_can_expose_a_cancellable_pair(gates, expected_gate):
    circuit = Circuit(ops=gates, num_qubits=3)
    outcome = search(circuit, mutations=0)
    assert outcome.best.ops == (expected_gate,)
    assert resource_score(outcome.best) == (1, 1)
    assert adjudicate_clifford_equivalence(circuit, outcome.best).verdict == "VALID"


def test_hadamard_y_hadamard_rewrite_is_certified_up_to_global_phase():
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("y", (0,)), Gate("h", (0,))),
                      num_qubits=1)
    rewrites = list(propose_rewrites(circuit))
    rewrite = next(candidate for rule, candidate in rewrites if rule == "h y h = y")
    assert rewrite.ops == (Gate("y", (0,)),)
    assert adjudicate_clifford_equivalence(circuit, rewrite).verdict == "VALID"


def test_the_self_inverse_set_is_exactly_the_gates_that_are_their_own_inverse():
    for name in SELF_INVERSE:
        # Two of them in a row is the identity: the empty circuit is equivalent.
        twice = Circuit(ops=(Gate(name, (0, 1) if name in ("cx", "cz", "swap")
                                  else (0,)),) * 2, num_qubits=2)
        empty = Circuit(ops=(), num_qubits=2)
        report = adjudicate_clifford_equivalence(empty, twice)
        assert report.verdict == "VALID", name


# ------------------------------------------------------------------- the workflow
def test_the_search_reaches_the_ground_truth_and_is_promoted():
    reference = junked_chain(12)
    outcome = search(reference, seed=1, mutations=3)
    # The reference is the 12-qubit GHZ chain plus cancelling junk, so the reachable
    # optimum is exactly the 12-gate chain.
    assert resource_score(outcome.best) == (12, 12)
    assert all(step.accepted for step in outcome.steps)
    assert all(step.verdict == "VALID" for step in outcome.steps)
    assert outcome.promoted is True
    assert outcome.verdict.tier == "stabilizer"
    assert outcome.verdict.claim_verdict == "MET"
    assert outcome.verdict.evidence == "SUFFICIENT"


def test_the_state_only_candidate_is_cheaper_and_still_refused():
    reference = junked_chain(12)
    star = Circuit(ops=(Gate("h", (0,)),
                        *[Gate("cx", (0, qubit)) for qubit in range(1, 12)]), num_qubits=12)
    assert star.resources()["operations"] < reference.resources()["operations"]
    unitary = adjudicate_clifford_equivalence(reference, star)
    state = adjudicate_clifford_equivalence(reference, star, on="state")
    assert unitary.verdict == "INVALID" and unitary.mismatched > 0
    assert state.verdict == "VALID"


def test_mutations_are_refused_with_the_image_named():
    outcome = search(junked_chain(8), seed=3, mutations=4)
    refused = [step for step in outcome.mutations if not step.accepted]
    assert refused, "the mutation channel must produce at least one refusal"
    for step in refused:
        assert step.verdict == "INVALID"
        assert step.mismatched > 0
        assert step.first_mismatch is not None
        assert step.first_mismatch["basis"]


# ---------------------------------------------------------------------- the record
def test_the_record_is_complete_and_passed_every_check(record):
    assert record["schema"] == "apedp.vqc.clifford_discovery.v1"
    assert record["qubits"] == 100
    for check in record["checks"]:
        assert check["passed"] is True, check
    assert record["verdict"].startswith("a 100-qubit Clifford circuit was rewritten")
    workload = record["workload"]
    # The junk cancels exactly: the reachable optimum of a 100-qubit GHZ chain is 100.
    assert workload["best_operations"] == 100
    assert workload["best_operations"] < workload["reference_operations"]
    assert workload["gates_saved"] == (workload["reference_operations"]
                                       - workload["best_operations"])
    # A truncated loop would look identical to a local optimum in the totals, so the
    # cap is asserted separately and the step count is required to be below it.
    assert len(record["steps"]) < workload["step_limit"]
    assert all(step["verdict"] == "VALID" and step["accepted"]
               for step in record["steps"])


def test_the_recorded_admission_is_evidence_sufficient(record):
    admission = record["admission"]
    assert admission["semantic_verdict"] == "VALID"
    assert admission["resource_verdict"] == "MET"
    assert admission["evidence_sufficiency"] == "SUFFICIENT"
    assert admission["promotion_decision"] == "PROMOTED"


def test_the_record_refuses_the_state_only_candidate(record):
    row = record["state_scope_only"]
    assert row["unitary_scope"] == "INVALID"
    assert row["unitary_mismatched"] > 0
    assert row["state_scope"] == "VALID"
    assert row["admission"]["promotable"] is False
    assert "not VALID" in row["refusal"]


def test_the_record_pins_the_code_that_produced_it(record):
    reproducibility = record["reproducibility"]
    assert reproducibility["script_sha256"] == hashlib.sha256(
        (ROOT / "scripts" / "measure_vqc_clifford_discovery.py").read_bytes()).hexdigest()
    for key, relative in (("search_sha256", "apedp_dyn/vqc/clifford_search.py"),
                          ("equivalence_sha256", "apedp_dyn/vqc/equivalence.py"),
                          ("discovery_sha256", "apedp_dyn/vqc/discovery.py")):
        assert reproducibility[key] == hashlib.sha256(
            (ROOT / relative).read_bytes()).hexdigest(), key


def test_the_recorded_limits_say_what_the_search_cannot_do(record):
    text = " ".join(record["limitations"]).lower()
    for topic in ("local rewrites", "greedy", "clifford", "noise"):
        assert topic in text, topic


def test_the_reference_is_recorded_as_deliberately_inefficient(record):
    """The 54.95% is against inserted work, and the record has to say so itself.

    Without this the headline reads as an advantage over an optimised GHZ
    preparation, which it is not: the discovered circuit *is* that preparation.
    """
    provenance = record["workload"]["reference_provenance"]
    assert provenance["deliberately_inefficient"] is True
    assert "cancelling" in provenance["described_as"]
    assert provenance["optimised_ghz_reference_operations"] == record["qubits"]
    assert provenance["optimised_ghz_reference_depth"] == record["qubits"]
    assert "does not establish" in provenance["what_the_reduction_measures"]
    assert "synthesiser" in provenance["what_it_does_not_measure"]
    # ... and the discovered circuit is exactly the optimised preparation.
    assert record["workload"]["best_operations"] == record["qubits"]
    assert record["workload"]["best_depth"] == record["qubits"]
    # The limitation list repeats it, so a reader who skips the workload block is
    # still told.
    assert any("deliberately inefficient" in line for line in record["limitations"])


def test_the_loop_admission_still_carries_the_four_separate_answers(record):
    """§6's second check: the safeguards are active in the loop, not only in tests."""
    admission = record["admission"]
    assert admission["semantic_verdict"] == "VALID"
    assert admission["resource_verdict"] == "MET"
    assert admission["evidence_sufficiency"] == "SUFFICIENT"
    assert admission["promotion_decision"] == "PROMOTED"
    assert admission["promotable"] is True
    # Every accepted step is certified VALID, and the star candidate that would win
    # on operations alone is refused by the same policy.
    assert all(step["verdict"] == "VALID" for step in record["steps"] if step["accepted"])
    assert record["state_scope_only"]["admission"]["promotable"] is False
