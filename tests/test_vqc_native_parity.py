"""Rust-first execution must agree with the Python reference, exactly.

The native engine (`rust/apedp-vqc`) is an optimisation, and an optimisation is
only allowed to be faster. These tests are the argument that it is also *the
same*: the derived conjugation tables, every Pauli image, every generator, every
exact distribution and the sampled counts are compared against the Python tier on
a frozen corpus, and the cases where the two must both refuse are checked too.

Four things are pinned beyond agreement:

* the fallback. With the extension absent every call returns ``None`` and the
  Python tier answers, so the package works without a Rust toolchain at all;
* the provenance. A result says which engine ran it, and ``engine="rust"``
  refuses rather than quietly downgrading -- a report cannot label a Python run
  native;
* the level of checking. Below the parity limit the referee re-derives the table
  in Python; above it the report says ``suite`` instead of implying a check that
  did not happen;
* the referee's independence: the engine produces images, Python writes the
  verdict.
"""
from __future__ import annotations

import math

import pytest

from apedp_dyn.vqc import native
from apedp_dyn.vqc.circuit import Circuit, Gate, Measure, Noise, Reset
from apedp_dyn.vqc.equivalence import (
    adjudicate_clifford_equivalence,
)
from apedp_dyn.vqc.stabilizer import (
    _PAIR,
    _PRODUCT,
    _SINGLE,
    CLIFFORD_GATES,
    StabilizerBackend,
    StabilizerState,
    StabilizerUnsupportedError,
    ghz_circuit,
)

pytestmark = pytest.mark.skipif(
    not native.is_available(),
    reason=f"the native engine is not built in this environment: {native.unavailable_reason()}",
)

ONE_QUBIT = ("h", "x", "y", "z", "s", "sdg")
TWO_QUBIT = ("cx", "cz", "swap")


def random_clifford_circuit(num_qubits: int, *, seed: int, length: int = 40) -> Circuit:
    """A fixed-seed Clifford circuit: the same gates for both engines, always."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ops: list = []
    for _ in range(length):
        if rng.random() < 0.5:
            ops.append(Gate(ONE_QUBIT[int(rng.integers(len(ONE_QUBIT)))],
                            (int(rng.integers(num_qubits)),)))
        else:
            pair = rng.choice(num_qubits, size=2, replace=False)
            ops.append(Gate(TWO_QUBIT[int(rng.integers(len(TWO_QUBIT)))],
                            (int(pair[0]), int(pair[1]))))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=0)


def _parity_corpus() -> list[Circuit]:
    circuits = [ghz_circuit(n, measure=False) for n in range(2, 9)]
    circuits.append(ghz_circuit(6, measure=False, basis="x"))
    circuits.append(bit_flip_corpus_circuit())
    circuits += [random_clifford_circuit(n, seed=seed)
                 for n, seed in ((3, 0), (4, 1), (5, 2), (6, 3), (7, 4))]
    return circuits


def bit_flip_corpus_circuit() -> Circuit:
    """The encoding ladder of the repetition code, without its measurements."""
    ops = [Gate("cx", (0, 1)), Gate("cx", (0, 2)), Gate("h", (2,)), Gate("s", (1,)),
           Gate("cx", (1, 2)), Gate("sdg", (0,)), Gate("cz", (0, 1))]
    return Circuit(ops=tuple(ops), num_qubits=3, num_bits=0)


def reset_feedforward_corpus_circuit() -> Circuit:
    """Mid-circuit reset followed by feedforward, with a deterministic truth table.

    The bits are *fixed by construction*, so agreement between the engines is a
    statement about composition order rather than a statistical coincidence:

    * ``bit 0`` is a fair coin, because qubit 0 was put in ``|+>``;
    * ``bit 1`` is always ``0`` -- qubit 1 is reset after the conditional ``X``, so
      whatever the feedforward did is erased before it is read;
    * ``bit 2`` equals ``bit 0`` -- the ``cx`` fires exactly when the ``x`` that set
      qubit 1 fired;
    * ``bit 3`` is a fair coin -- an ``H`` then a ``Z`` under a two-bit conjunction
      maps ``|0>`` to ``|->``, whose readout is uniform in both branches.
    """
    ops = [
        Gate("h", (0,)),
        Measure(0, 0),
        Gate("x", (1,), condition=((0, True),)),
        Reset(1),
        Measure(1, 1),
        # The control is qubit 0, which holds bit 0 after its measurement: the reset
        # above cleared qubit 1, so a `cx` controlled on it would be a no-op and would
        # test nothing. That mistake was in this fixture first, not in an engine.
        Gate("cx", (0, 2), condition=((0, True),)),
        Measure(2, 2),
        Reset(2),
        Gate("h", (3,)),
        Gate("z", (3,), condition=((0, True), (2, True))),
        Measure(3, 3),
    ]
    return Circuit(ops=tuple(ops), num_qubits=4, num_bits=4)


def polarity_corpus_circuit() -> Circuit:
    """Both condition polarities, on the same measured bit, in one circuit.

    ``x`` on qubit 1 fires when bit 0 is ``1``; ``x`` on qubit 2 fires when bit 0 is
    ``0``. Two engines that agree on one polarity and disagree on the other is the
    failure this circuit exists to make visible, and the readout is deterministic in
    both branches, so no shot budget can blur it. (A condition on a bit that is never
    written cannot be expressed at all: the IR refuses it as "read before it is
    written", upstream of either engine.)
    """
    ops = [
        Gate("h", (0,)),
        Measure(0, 0),
        Gate("x", (1,), condition=((0, True),)),
        Gate("x", (2,), condition=((0, False),)),
        Measure(1, 1),
        Measure(2, 2),
    ]
    return Circuit(ops=tuple(ops), num_qubits=3, num_bits=3)


def teleportation_corpus_circuit() -> Circuit:
    """Bell pair, Alice's measurement, Bob's conditional corrections, readout."""
    ops = [
        Gate("h", (0,)),                 # the message is |+>, so the X-basis
        Gate("h", (1,)), Gate("cx", (1, 2)),  # readout of qubit 2 is decided
        Gate("cx", (0, 1)), Gate("h", (0,)),
        Measure(0, 0), Measure(1, 1),
        Gate("x", (2,), condition=((1, True),)),
        Gate("z", (2,), condition=((0, True),)),
        Gate("h", (2,)), Measure(2, 2),
    ]
    return Circuit(ops=tuple(ops), num_qubits=3, num_bits=3)


# ------------------------------------------------------------- the tables themselves
def test_the_native_conjugation_tables_are_the_python_tiers_tables():
    """Not "close": the derived tables are compared entry by entry, as integers.

    Both sides derive these tables from the same gate matrices, so a single sign
    that differs is a real disagreement rather than a rounding artefact, and this
    test is what makes the derivation checkable instead of merely parallel.
    """
    tables = native.gate_tables()
    assert tables is not None
    for name in ONE_QUBIT:
        assert [tuple(rule) for rule in tables["single"][name]] == list(_SINGLE[name]), name
    for name in TWO_QUBIT:
        expected = [_PAIR[name][(first, second)]
                    for first in range(4) for second in range(4)]
        assert [tuple(rule) for rule in tables["pair"][name]] == expected, name
    product = _PRODUCT
    assert set(tables["single"]) == set(ONE_QUBIT)
    assert set(tables["pair"]) == set(TWO_QUBIT)
    assert product[(1, 2)] == (1, 3)  # X Y = iZ, the one identity the tables sit on


def test_the_clifford_gate_set_is_the_same_in_both_languages():
    """The native module advertises its set; it must be the tier's set, not a superset."""
    import apedp_vqc

    assert set(apedp_vqc.CLIFFORD_GATES) == set(CLIFFORD_GATES)


# ------------------------------------------------------------- the frozen corpus
@pytest.mark.parametrize("index", range(len(_parity_corpus())))
def test_every_circuit_in_the_corpus_agrees_image_for_image(index: int):
    circuit = _parity_corpus()[index]
    summary = native.assert_parity(circuit)
    assert summary["images_compared"] == 2 * circuit.num_qubits
    assert summary["generators_compared"] == circuit.num_qubits


def test_the_exact_distribution_agrees_on_every_small_corpus_circuit():
    for circuit in _parity_corpus():
        if circuit.num_qubits > 8:
            continue
        native_distribution = native.distribution(circuit)
        assert native_distribution is not None
        state = StabilizerState(circuit.num_qubits)
        for op in circuit.ops:
            state.apply_gate(op.name, op.targets, op.params)
        reference = state.distribution()
        assert native_distribution == reference, circuit.sha256()
        assert sum(native_distribution.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("num_qubits", (6, 9, 11))
def test_the_generators_agree_at_widths_where_the_dense_tier_cannot_go(num_qubits: int):
    circuit = ghz_circuit(num_qubits, measure=False)
    state = StabilizerState(num_qubits)
    for op in circuit.ops:
        state.apply_gate(op.name, op.targets, op.params)
    assert native.generators(circuit) == state.generator_strings()
    assert native.expectation(circuit, "X" * num_qubits) == 1.0


def test_a_relative_phase_is_seen_identically_by_both_engines():
    """`x s x sdg` is `x` times a relative phase; both engines must say so."""
    plain = Circuit(ops=(Gate("x", (0,)),), num_qubits=1, num_bits=0)
    relative = Circuit(ops=(Gate("x", (0,)), Gate("s", (0,)), Gate("x", (0,)),
                            Gate("sdg", (0,))), num_qubits=1, num_bits=0)
    native_plain = native.images(plain)
    native_relative = native.images(relative)
    assert native_plain is not None and native_relative is not None
    assert native_plain != native_relative
    # The reference tier sees exactly the same disagreement, in the same rows.
    from apedp_dyn.vqc.stabilizer import clifford_pauli_images
    assert native_plain == clifford_pauli_images(plain)
    assert native_relative == clifford_pauli_images(relative)
    report = adjudicate_clifford_equivalence(plain, relative)
    assert report.verdict == "INVALID"
    assert report.mismatched == 2 and report.compared == 2


def test_a_global_phase_is_invisible_to_both_engines():
    """`s sdg` is the identity times -1, which no image and no verdict can see."""
    identity = Circuit(ops=(), num_qubits=1, num_bits=0)
    shifted = Circuit(ops=(Gate("s", (0,)), Gate("sdg", (0,))), num_qubits=1, num_bits=0)
    report = adjudicate_clifford_equivalence(identity, shifted)
    assert report.verdict == "VALID" and report.mismatched == 0
    assert native.images(identity) == native.images(shifted)


def test_the_sampler_reproduces_the_exact_distribution_it_claims():
    """Rust samples, so the check is statistical: 2000 shots against exact weights."""
    circuit = ghz_circuit(4)
    shots = 2000
    result = StabilizerBackend().run(circuit, shots=shots, seed=17)
    assert result.engine == "rust"
    state = StabilizerState(4)
    for op in circuit.ops:
        if isinstance(op, Gate):  # the ladder, before the measurements
            state.apply_gate(op.name, op.targets, op.params)
    for outcome in ("0000", "1111"):
        expected = state.probability_of_bitstring(tuple(bit == "1" for bit in outcome))
        assert expected == pytest.approx(0.5, abs=1e-12)
        observed = result.counts.get(outcome, 0) / shots
        assert abs(observed - expected) <= 5 * math.sqrt(expected * (1 - expected) / shots)


def _sampler_corpus() -> list[Circuit]:
    """The measured circuits both engines execute: conditions, feedforward, resets."""
    return [teleportation_corpus_circuit(), reset_feedforward_corpus_circuit(),
            polarity_corpus_circuit()]


@pytest.mark.parametrize("index", range(len(_sampler_corpus())))
def test_the_sampler_corpus_reproduces_its_deterministic_truth_table(index: int):
    """The outcomes a measured circuit can produce are decided by the circuit.

    Both engines are run on every circuit in the sampler corpus and both must produce
    only the outcomes the construction allows: a reset that is not applied, a condition
    read in the wrong order or a bit that defaults to ``1`` changes the *set* of
    outcomes, which no amount of shots can hide.
    """
    circuit = _sampler_corpus()[index]
    allowed = {
        0: {"000", "010", "100", "110"},          # teleportation: bit 2 is the message
        1: {"0000", "0001", "1010", "1011"},       # bit 1 reset, bit 2 copies bit 0, bit 3 free
        2: {"001", "110"},                          # bit 1 copies bit 0, bit 2 negates it
    }[index]
    for engine in ("rust", "python"):
        result = StabilizerBackend().run(circuit, shots=256, seed=11, engine=engine)
        assert result.engine == engine
        assert set(result.counts) == allowed, (engine, set(result.counts))


def test_the_reset_feedforward_circuit_separates_the_three_feature_kinds():
    """Feedforward, reset and a two-qubit condition, each with its own observable."""
    circuit = reset_feedforward_corpus_circuit()
    for engine in ("rust", "python"):
        result = StabilizerBackend().run(circuit, shots=512, seed=3, engine=engine)
        ones = sum(1 for row in result.samples if row[0])
        # bit 0 is a fair coin (the H), not a decided value.
        assert abs(ones / len(result.samples) - 0.5) <= 0.1, (engine, ones)
        # bit 1 is the reset: always 0, in every shot of every seed.
        assert not any(row[1] for row in result.samples), engine
        # bit 2 copies bit 0 exactly -- that is the conditioned `cx` doing its job.
        assert all(row[2] == row[0] for row in result.samples), engine


def test_a_conditioned_gate_is_refused_by_the_image_parity_check():
    """A classically conditioned gate is not a unitary, so it has no image table."""
    circuit = Circuit(ops=(Gate("x", (0,), condition=((0, True),)),
                           Gate("cx", (0, 1), condition=((1, True),))),
                      num_qubits=2, num_bits=2)
    with pytest.raises(ValueError, match="conditioned_gate"):
        native.assert_parity(circuit)
    # And the tier's own conjugation routine refuses the same input, by name.
    from apedp_dyn.vqc.stabilizer import conjugate_pauli
    with pytest.raises(Exception, match="unconditioned"):
        conjugate_pauli((1, 0), 0, ops=circuit.ops)


def test_a_multi_control_gate_is_refused_by_both_engines():
    """`ccx` is outside the Clifford set: neither engine executes it, and neither guesses."""
    from apedp_dyn.vqc.stabilizer import StabilizerUnsupportedError

    circuit = Circuit(ops=(Gate("h", (0,)), Gate("ccx", (0, 1, 2))), num_qubits=3,
                      num_bits=0)
    with pytest.raises(StabilizerUnsupportedError):
        StabilizerBackend().run(circuit, shots=8, seed=1, engine="rust")
    with pytest.raises(StabilizerUnsupportedError):
        StabilizerBackend().run(circuit, shots=8, seed=1, engine="python")


def test_the_conditional_teleportation_sampler_matches_the_reference():
    circuit = teleportation_corpus_circuit()
    shots = 2000
    native_result = StabilizerBackend().run(circuit, shots=shots, seed=23, engine="rust")
    reference_result = StabilizerBackend().run(circuit, shots=shots, seed=23, engine="python")
    assert native_result.engine == "rust" and reference_result.engine == "python"
    assert set(native_result.counts) == set(reference_result.counts) == {
        "000", "010", "100", "110"}
    for count in native_result.counts.values():
        assert abs(count - shots / 4) <= 5 * math.sqrt(shots * 0.1875)


def test_the_same_seed_replays_and_a_different_seed_does_not():
    circuit = ghz_circuit(6)
    first = StabilizerBackend().run(circuit, shots=64, seed=5)
    second = StabilizerBackend().run(circuit, shots=64, seed=5)
    third = StabilizerBackend().run(circuit, shots=64, seed=6)
    assert first.counts == second.counts and first.samples == second.samples
    assert third.samples != first.samples
    assert first.seed == 5 and first.as_dict()["engine"] == "rust"


# ------------------------------------------------------------- the fallback, and its honesty
def test_the_bridge_is_inert_when_the_extension_is_absent(monkeypatch):
    monkeypatch.setattr(native, "_MODULE", None)
    assert not native.is_available()
    assert "not importable" in native.unavailable_reason()
    circuit = ghz_circuit(3, measure=False)
    assert native.images(circuit) is None
    assert native.distribution(circuit) is None
    assert native.generators(circuit) is None
    assert native.expectation(circuit, "XXX") is None
    assert native.run(circuit, shots=8, seed=1) is None


def test_the_python_tier_answers_every_call_without_the_extension(monkeypatch):
    monkeypatch.setattr(native, "_MODULE", None)
    circuit = ghz_circuit(4)
    result = StabilizerBackend().run(circuit, shots=200, seed=9)
    assert result.engine == "python"
    assert sum(result.counts.values()) == 200

    reference = ghz_circuit(4, measure=False)
    report = adjudicate_clifford_equivalence(reference, reference)
    assert report.verdict == "VALID"
    assert report.image_engine == "python" and report.image_verified_by == "python"


def test_a_requested_native_run_refuses_rather_than_downgrading(monkeypatch):
    monkeypatch.setattr(native, "_MODULE", None)
    with pytest.raises(StabilizerUnsupportedError, match="native engine could not run"):
        StabilizerBackend().run(ghz_circuit(3), shots=8, seed=1, engine="rust")


def test_the_engine_argument_is_validated():
    with pytest.raises(ValueError, match="engine must be"):
        StabilizerBackend().run(ghz_circuit(3), shots=8, seed=1, engine="gpu")
    with pytest.raises(ValueError, match="engine must be"):
        adjudicate_clifford_equivalence(ghz_circuit(3, measure=False),
                                        ghz_circuit(3, measure=False), engine="gpu")


def test_the_python_only_pin_forces_the_reference_path(monkeypatch):
    monkeypatch.setenv("APEDP_VQC_PYTHON_ONLY", "1")
    assert native.python_only() and not native.is_available()
    assert native.images(ghz_circuit(3, measure=False)) is None
    assert StabilizerBackend().run(ghz_circuit(3), shots=8, seed=1).engine == "python"


# ------------------------------------------------------------- refusals, both ways
def test_both_engines_refuse_a_non_clifford_gate():
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("t", (0,))), num_qubits=1, num_bits=0)
    assert native.images(circuit) is None            # the fast path declines
    assert native.last_failure() is not None
    report = adjudicate_clifford_equivalence(circuit, circuit)
    assert report.verdict == "INVALID"
    assert any("'t' is not Clifford" in str(failure) for failure in
               [report.failed(), [check.as_dict() for check in report.checks]])
    assert native.run(circuit, shots=4, seed=1) is None
    with pytest.raises(StabilizerUnsupportedError, match="not in this tier's Clifford set"):
        StabilizerBackend().run(circuit, shots=4, seed=1)


def test_a_noise_operation_is_refused_by_name_and_not_simulated():
    noisy = Circuit(ops=(Gate("h", (0,)), Noise(0, "depolarizing", 0.1)),
                    num_qubits=1, num_bits=0)
    assert native.run(noisy, shots=4, seed=1) is None
    with pytest.raises(StabilizerUnsupportedError, match="statevector tier"):
        StabilizerBackend().run(noisy, shots=4, seed=1)


def test_a_measured_circuit_has_no_image_table_in_either_engine():
    circuit = Circuit(ops=(Gate("h", (0,)), Measure(0, 0)), num_qubits=1, num_bits=1)
    assert native.images(circuit) is None
    report = adjudicate_clifford_equivalence(circuit, circuit)
    assert report.verdict == "INVALID"
    reason = " ".join(str(check.as_dict()["measured"]) for check in report.checks)
    assert "measure on qubit" in reason


def test_the_parity_helper_insists_on_a_unitary_circuit():
    with pytest.raises(ValueError, match="measurement-free"):
        native.assert_parity(ghz_circuit(3))


# ------------------------------------------------------------- the referee stays in Python
def test_the_referee_re_derives_the_table_in_python_below_the_parity_limit():
    circuit = random_clifford_circuit(8, seed=11)
    report = adjudicate_clifford_equivalence(circuit, circuit)
    assert report.verdict == "VALID"
    assert report.image_engine == "rust"
    assert report.image_verified_by == "python"
    assert report.as_dict()["image_verified_by"] == "python"


def test_above_the_parity_limit_the_report_says_suite_not_verified():
    wide = ghz_circuit(native.PARITY_LIMIT + 2, measure=False)
    report = adjudicate_clifford_equivalence(wide, wide)
    assert report.verdict == "VALID"
    assert report.image_engine == "rust"
    assert report.image_verified_by == "suite"
    assert report.compared == 2 * (native.PARITY_LIMIT + 2)


def test_a_corrupted_native_table_is_refused_rather_than_adjudicated(monkeypatch):
    circuit = ghz_circuit(3, measure=False)
    good = native.images(circuit)
    assert good is not None
    corrupted = list(good)
    labels, phase = corrupted[0]
    corrupted[0] = (labels, (phase + 2) % 4)
    monkeypatch.setattr(native, "images", lambda _circuit: tuple(corrupted))
    with pytest.raises(native.NativeParityError, match="disagrees with the reference"):
        adjudicate_clifford_equivalence(circuit, circuit)


def test_a_hundred_qubit_chain_is_the_same_answer_in_both_engines():
    """The flagship width: closed-form generators and correlators, both paths."""
    circuit = ghz_circuit(100, measure=False)
    generators = native.generators(circuit)
    assert generators is not None and len(generators) == 100
    assert generators[0] == "+" + "X" * 100
    assert generators[1] == "+ZZ" + "I" * 98
    assert native.expectation(circuit, "X" * 100) == 1.0
    labels = ["I"] * 100
    labels[0] = labels[99] = "Z"
    assert native.expectation(circuit, "".join(labels)) == 1.0

    measured = ghz_circuit(100)
    result = StabilizerBackend().run(measured, shots=64, seed=31)
    assert result.engine == "rust"
    assert set(result.counts) <= {"0" * 100, "1" * 100}
    assert sum(result.counts.values()) == 64


def test_the_counters_record_which_path_answered():
    native.reset_tallies()
    circuit = ghz_circuit(3, measure=False)
    native.images(circuit)
    native.distribution(circuit)
    native.assert_parity(circuit)
    counts = native.tallies()
    assert counts["rust"] >= 4 and counts["fallback"] == 0
    assert native.reset_tallies() is None
    assert native.tallies() == {"rust": 0, "python": 0, "fallback": 0}
