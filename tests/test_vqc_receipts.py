"""The adjudication receipt: what it passes, what it refuses, and what it admits.

``apedp_dyn/vqc/receipts.py`` is the module that decides whether a quantum
claim is allowed to stand, and it was on
``benchmarks/untested_modules_register.json`` until 2026-11-17: no test named
it. The audit that added these tests found one hole, pinned below: an outcome
with a declared probability of zero was not scored at all, so forbidden strings
could appear without the consistency check noticing.
"""
from __future__ import annotations

import json

import pytest

from apedp_dyn.vqc.backends import RunResult
from apedp_dyn.vqc.circuit import Circuit, Gate, Measure
from apedp_dyn.vqc.receipts import adjudicate


def bell() -> Circuit:
    return Circuit(
        ops=(Gate("h", (0,)), Gate("cx", (0, 1)), Measure(0, 0), Measure(1, 1)),
        num_qubits=2,
        num_bits=2,
    )


class _FixedBackend:
    """A backend that reports exactly the distribution a test wants to judge."""

    name = "fixed"

    def __init__(self, counts: dict[str, int], ancilla_excitation: dict[int, float] | None = None):
        self.counts = counts
        self.ancilla_excitation = ancilla_excitation or {}

    def run(self, circuit: Circuit, *, shots: int, seed: int | None = None) -> RunResult:
        return RunResult(
            backend=self.name,
            shots=sum(self.counts.values()),
            seed=seed,
            counts=dict(self.counts),
            samples=(),
            ancilla_excitation=dict(self.ancilla_excitation),
        )


def test_a_circuit_that_does_what_it_claims_is_valid():
    receipt = adjudicate(bell(), expected={"00": 0.5, "11": 0.5}, shots=4096, seed=1)
    assert receipt.verdict == "VALID"
    assert [check.passed for check in receipt.checks] == [True, True, True]
    assert receipt.reproducibility["shots"] == 4096
    assert receipt.identity["circuit_sha256"] == bell().sha256()
    json.dumps(receipt.to_json() if hasattr(receipt, "to_json") else receipt.__dict__, default=str)


def test_a_wrong_expectation_is_invalid():
    receipt = adjudicate(bell(), expected={"00": 0.5, "01": 0.5}, shots=4096, seed=1)
    assert receipt.verdict == "INVALID"


def test_too_few_shots_is_unresolved_not_a_pass():
    """The third verdict: nothing failed, nothing is established."""
    receipt = adjudicate(bell(), expected={"00": 0.5, "11": 0.5}, shots=4, seed=1, min_effect=0.01)
    assert receipt.verdict == "UNRESOLVED"
    power = next(check for check in receipt.checks if "power" in check.name)
    assert power.passed is None
    assert power.measured["required_shots"] > 4


def test_a_forbidden_outcome_fails_even_when_the_declared_ones_look_right():
    """The hole this file was written for.

    The expectation declares four outcomes at 0.25 and says nothing else can
    happen. The backend reports a fifth string on 2 % of shots and the declared
    four scaled down to make room. Before 2026-11-17 the fifth outcome was not
    scored (only ``want > 0`` outcomes were), the four declared ones stayed
    within four standard errors, and the receipt read VALID.
    """
    counts = {"000": 2450, "001": 2450, "010": 2450, "011": 2450, "111": 200}
    circuit = Circuit(
        ops=(Gate("h", (0,)), Gate("h", (1,)), Measure(0, 0), Measure(1, 1), Measure(2, 2)),
        num_qubits=3,
        num_bits=3,
    )
    expected = {"000": 0.25, "001": 0.25, "010": 0.25, "011": 0.25}
    receipt = adjudicate(circuit, expected=expected, backend=_FixedBackend(counts), shots=sum(counts.values()))
    consistency = next(check for check in receipt.checks if "distribution" in check.name)
    assert consistency.measured["worst_outcome"] == "111"
    assert receipt.verdict == "INVALID"
    # The declared outcomes alone would not have failed: each is 0.245 against
    # 0.25, which is about 1.2 standard errors at 10,000 shots.
    assert all(abs(consistency.measured["observed"][key] - 0.25) < 0.01 for key in expected)


def test_a_declared_certainty_that_did_not_happen_fails():
    counts = {"0": 900, "1": 100}
    circuit = Circuit(ops=(Measure(0, 0),), num_qubits=1, num_bits=1)
    receipt = adjudicate(circuit, expected={"0": 1.0}, backend=_FixedBackend(counts), shots=1000)
    assert receipt.verdict == "INVALID"


def test_the_receipt_reports_its_own_family_wise_alarm_rate():
    """The gate is per-outcome, so the reader is told what that costs over many
    outcomes rather than being left to assume sigma is the whole story."""
    receipt = adjudicate(bell(), expected={"00": 0.5, "11": 0.5}, shots=4096, seed=1)
    consistency = next(check for check in receipt.checks if "distribution" in check.name)
    assert consistency.measured["scored_outcomes"] == 2
    per_outcome = consistency.measured["per_outcome_false_alarm_rate"]
    assert per_outcome == pytest.approx(6.334e-05, rel=1e-3)
    assert consistency.measured["familywise_false_alarm_rate"] == pytest.approx(
        1.0 - (1.0 - per_outcome) ** 2
    )


def test_a_dirty_ancilla_fails_whatever_the_distribution_says():
    counts = {"00": 5000, "11": 5000}
    circuit = Circuit(
        ops=(Gate("h", (0,)), Gate("cx", (0, 1)), Measure(0, 0), Measure(1, 1)),
        num_qubits=3,
        num_bits=2,
        ancillas=(2,),
    )
    backend = _FixedBackend(counts, ancilla_excitation={2: 0.4})
    receipt = adjudicate(circuit, expected={"00": 0.5, "11": 0.5}, backend=backend, shots=10000)
    assert receipt.verdict == "INVALID"
    assert receipt.ancillas["worst_excitation"] == 0.4


def test_an_invalid_circuit_is_refused_before_it_is_run():
    circuit = Circuit(ops=(Gate("cx", (0, 5)),), num_qubits=2)
    receipt = adjudicate(circuit, expected={"00": 1.0})
    assert receipt.verdict == "INVALID"
    assert receipt.checks[0].passed is False
    assert receipt.ancillas["checked"] is False


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        adjudicate(bell(), expected={"00": 1.0}, shots=0)
    with pytest.raises(ValueError):
        adjudicate(bell(), expected={"00": 1.0}, sigma=0.0)
