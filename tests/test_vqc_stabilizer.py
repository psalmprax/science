"""The stabilizer tier, checked against the dense tier and against closed forms.

Two kinds of check live here:

* **agreement** -- the same circuit builder, both tiers, every outcome probability
  and a set of Pauli expectations compared exactly, for chain lengths the dense
  tier can hold;
* **reach** -- a 100-qubit GHZ chain, where the dense tier has nothing to say
  (2**100 amplitudes), checked against the closed-form GHZ predictions the dense
  tier has just confirmed at small `n`.

The second is only as good as the first, which is why both are in this file and
use the same `ghz_circuit` function.
"""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

import numpy as np
import pytest

from apedp_dyn.vqc import Circuit, Gate, Measure, Noise, StatevectorBackend
from apedp_dyn.vqc.gates import matrix_for
from apedp_dyn.vqc.stabilizer import (
    CLIFFORD_GATES,
    StabilizerBackend,
    StabilizerState,
    StabilizerUnsupportedError,
    ghz_circuit,
    ghz_even_parity_probability,
    ghz_generators,
    is_clifford_circuit,
)

LABELS = "IXYZ"
_PAULI_INDEX = {"x": 1, "y": 2, "z": 3}


# --------------------------------------------------------------------- helpers
def dense_state(circuit: Circuit) -> np.ndarray:
    """The pure statevector of an unmeasured circuit on the dense tier."""
    result = StatevectorBackend().run(circuit, shots=1, seed=0, keep_states=True)
    return result.final_states[0]


def dense_probabilities(circuit: Circuit) -> np.ndarray:
    return np.abs(dense_state(circuit)) ** 2


def stabilizer_state(circuit: Circuit) -> StabilizerState:
    """The state after the unitary part of `circuit`, by the gate conjugation path."""
    state = StabilizerState(circuit.num_qubits)
    for op in circuit.ops:
        assert isinstance(op, Gate), "the unitary prefix must contain no measurements"
        state.apply_gate(op.name, op.targets, op.params)
    return state


def bits_of(index: int, num_qubits: int) -> tuple[bool, ...]:
    """Qubit `q` is bit `q` of the amplitude index, which is the IR's convention."""
    return tuple(bool((index >> qubit) & 1) for qubit in range(num_qubits))


def dense_expectation(state: np.ndarray, labels: Sequence[int]) -> float:
    """`<psi|P|psi>` computed by applying the Pauli to a copy of the state."""
    from apedp_dyn.quantum_compilation.statevector_sim import StatevectorSimulator

    simulator = StatevectorSimulator(int(math.log2(state.size)))
    simulator.state = np.array(state, dtype=np.complex128)
    for qubit, label in enumerate(labels):
        if label:
            simulator.apply_matrix(qubit, matrix_for(LABELS[label].lower()))
    return float(np.real(np.vdot(state, simulator.state)))


def random_clifford_circuit(num_qubits: int, *, seed: int, depth: int = 12) -> Circuit:
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


# ------------------------------------------------------- agreement with the dense tier
def test_a_bell_pair_has_the_textbook_generators():
    state = stabilizer_state(ghz_circuit(2, measure=False))
    assert set(state.generator_strings()) == {"+XX", "+ZZ"}


@pytest.mark.parametrize("num_qubits", range(2, 13))
def test_every_ghz_probability_matches_the_dense_tier(num_qubits: int):
    circuit = ghz_circuit(num_qubits, measure=False)
    state = stabilizer_state(circuit)
    dense = dense_probabilities(circuit)

    total = 0.0
    for index in range(1 << num_qubits):
        probability = state.probability_of_bitstring(bits_of(index, num_qubits))
        assert probability == pytest.approx(dense[index], abs=1e-12), (
            f"n={num_qubits} outcome {bits_of(index, num_qubits)}")
        total += probability
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_random_clifford_circuits_agree_with_the_dense_tier(seed: int):
    num_qubits = 4
    circuit = random_clifford_circuit(num_qubits, seed=seed)
    state = stabilizer_state(circuit)
    dense = dense_probabilities(circuit)

    for index in range(1 << num_qubits):
        probability = state.probability_of_bitstring(bits_of(index, num_qubits))
        assert probability == pytest.approx(dense[index], abs=1e-12)

    rng = random.Random(seed + 1000)
    for _ in range(8):
        labels = tuple(rng.randrange(4) for _ in range(num_qubits))
        assert state.expectation(labels) == pytest.approx(
            dense_expectation(dense_state(circuit), labels), abs=1e-12)


def test_the_generators_are_a_consistent_stabilizer_state():
    """Every generator must have expectation +1, and the group must be independent."""
    circuit = random_clifford_circuit(5, seed=7, depth=20)
    state = stabilizer_state(circuit)
    assert len(state.generator_strings()) == 5
    for index, labels in enumerate(state.generators):
        # A generator's own expectation is its phase: `-ZIIIX` is a stabilizer of
        # magnitude one with a minus sign, and it must be *exactly* that sign.
        phase = state.phases[index] % 4
        assert phase in (0, 2), f"generator {state.generator_strings()[index]} has an i"
        magnitude = 1.0 if phase == 0 else -1.0
        assert state.expectation(labels) == magnitude
        assert state.stabilizes(labels, sign=0 if magnitude == 1.0 else 1)
    # Independence: `n` independent commuting rows leave exactly one state, so the
    # distribution must be normalised and the dense tier must agree with it.
    assert sum(state.distribution().values()) == pytest.approx(1.0, abs=1e-12)


def test_sampled_counts_match_the_dense_tier_in_distribution():
    circuit = random_clifford_circuit(3, seed=11)
    measured = Circuit(
        ops=circuit.ops + tuple(Measure(qubit, qubit) for qubit in range(3)),
        num_qubits=3,
        num_bits=3,
    )
    shots = 4000
    fast = StabilizerBackend().run(measured, shots=shots, seed=3)
    dense = StatevectorBackend().run(measured, shots=shots, seed=3)

    keys = set(fast.counts) | set(dense.counts)
    for key in keys:
        assert abs(fast.counts.get(key, 0) - dense.counts.get(key, 0)) <= 4 * math.sqrt(
            0.25 * shots) + 10, key


# --------------------------------------------------- a hundred qubits, and why it counts
def test_a_hundred_qubit_ghz_chain_has_the_closed_form_generators():
    circuit = ghz_circuit(100, measure=False)
    state = stabilizer_state(circuit)
    assert set(state.generators) == set(ghz_generators(100))
    assert len(state.generators) == 100
    # The two stabilizers the paper would write down, verbatim.
    assert state.stabilizes(tuple([1] * 100))
    assert state.expectation(tuple([1] * 100)) == 1.0


def test_a_hundred_qubit_ghz_chain_has_the_closed_form_distribution():
    state = stabilizer_state(ghz_circuit(100, measure=False))
    ground = tuple([False] * 100)
    excited = tuple([True] * 100)

    assert state.probability_of_bitstring(ground) == 0.5
    assert state.probability_of_bitstring(excited) == 0.5
    # Every other string is impossible, including neighbours one bit away.
    assert state.probability_of_bitstring((True, *ground[1:])) == 0.0
    assert state.probability_of_bitstring((*ground[:50], True, *ground[51:])) == 0.0
    assert state.probability_of_bitstring((*excited[:-1], False)) == 0.0


def test_the_hundred_qubit_run_produces_only_the_two_ghz_outcomes():
    result = StabilizerBackend().run(ghz_circuit(100), shots=128, seed=5)
    assert set(result.counts) == {"0" * 100, "1" * 100}
    assert result.counts["0" * 100] + result.counts["1" * 100] == 128
    # Seeded 50/50: 128 shots, and the split is inside five sigma of the mean.
    for count in result.counts.values():
        assert abs(count - 64) <= 5 * math.sqrt(128 * 0.25)


def test_pairwise_parities_at_a_hundred_qubits_are_perfectly_correlated():
    state = stabilizer_state(ghz_circuit(100, measure=False))
    pairs = [(0, 99), (0, 50), (17, 83), (48, 49)]
    for first, second in pairs:
        labels = [0] * 100
        labels[first] = labels[second] = 3
        assert state.expectation(tuple(labels)) == 1.0

    # ... and the single-qubit and distant-X correlators vanish, as GHZ requires.
    for first, second in pairs:
        single = [0] * 100
        single[first] = 1
        assert state.expectation(tuple(single)) == 0.0
        single[second] = 1
        assert state.expectation(tuple(single)) == 0.0


def test_the_x_basis_chain_is_uniform_over_even_parity_strings():
    circuit = ghz_circuit(100, measure=True, basis="x")
    result = StabilizerBackend().run(circuit, shots=64, seed=9)
    assert all(bits.count("1") % 2 == 0 for bits in result.counts)
    assert len(result.counts) == 64, "2**100 strings, so 64 shots must not collide"
    state = stabilizer_state(ghz_circuit(100, measure=False, basis="x"))
    assert state.probability_of_bitstring(tuple([False] * 100)) == pytest.approx(
        ghz_even_parity_probability(100), abs=0.0)
    assert state.probability_of_bitstring((True, *tuple([False] * 99))) == 0.0


@pytest.mark.parametrize("num_qubits", (3, 4, 5, 6, 7))
def test_the_dense_tier_confirms_the_construction_the_hundred_qubit_run_extends(
        num_qubits: int):
    """The 100-qubit circuit is the same ladder, gate for gate, just longer."""
    small = ghz_circuit(num_qubits, measure=False)
    large = ghz_circuit(100, measure=False)
    assert large.ops[:len(small.ops)] == small.ops
    assert large.ops[len(small.ops) - 1].targets == (num_qubits - 2, num_qubits - 1)

    dense = dense_probabilities(small)
    state = stabilizer_state(small)
    for index in range(1 << num_qubits):
        assert state.probability_of_bitstring(bits_of(index, num_qubits)) == pytest.approx(
            dense[index], abs=1e-12)


# --------------------------------------------------------------------- domain edges
def test_non_clifford_and_noisy_circuits_are_refused_by_name():
    t_gate = Circuit(ops=(Gate("t", (0,)),), num_qubits=1)
    with pytest.raises(StabilizerUnsupportedError, match="not in this tier"):
        StabilizerBackend().run(t_gate, shots=1)

    rotation = Circuit(ops=(Gate("rz", (0,), (0.3,)),), num_qubits=1)
    with pytest.raises(StabilizerUnsupportedError, match="not in this tier"):
        StabilizerBackend().run(rotation, shots=1)

    noisy = Circuit(ops=(Gate("h", (0,)),
                         Noise(0, "depolarizing", 0.1),
                         Measure(0, 0)), num_qubits=1, num_bits=1)
    with pytest.raises(StabilizerUnsupportedError, match="mixture"):
        StabilizerBackend().run(noisy, shots=1)

    with pytest.raises(StabilizerUnsupportedError, match="amplitudes"):
        StabilizerBackend().run(ghz_circuit(3), shots=1, keep_states=True)


def test_is_clifford_circuit_classifies_the_boundary():
    assert is_clifford_circuit(ghz_circuit(5))
    assert is_clifford_circuit(Circuit(ops=(Gate("ccx", (0, 1, 2)),), num_qubits=3)) is False
    assert is_clifford_circuit(Circuit(ops=(Gate("sdg", (0,)),), num_qubits=1))


def test_a_valid_tableau_validates_and_the_gate_path_keeps_it_valid():
    """`validate()` is the structural check the arithmetic cannot make on its own."""
    for seed in (0, 1, 2):
        state = stabilizer_state(random_clifford_circuit(6, seed=seed, depth=25))
        state.validate()
    # Measurement branches and a 100-qubit state must stay valid too.
    ghz = stabilizer_state(ghz_circuit(100, measure=False))
    ghz.validate()
    branch = stabilizer_state(ghz_circuit(4, measure=False))
    branch.measure(2)
    branch.validate()
    assert len(ghz.generators) == 100


def test_anticommuting_generators_are_refused_not_averaged():
    """`X Z` and `Z I`: there is no simultaneous eigenspace, so there is no state."""
    state = StabilizerState(2)
    state.generators[0] = (1, 3)   # XZ
    state.generators[1] = (3, 0)   # ZI -- anticommutes with XZ on qubit 0
    with pytest.raises(StabilizerUnsupportedError, match="anticommute"):
        state.validate()


def test_a_rank_deficient_tableau_is_refused():
    """Two copies of one generator span a rank-1 group: a mixed state, not this type."""
    state = StabilizerState(2)
    state.generators[1] = state.generators[0]
    with pytest.raises(StabilizerUnsupportedError, match="not pure"):
        state.validate()


def test_a_repeated_measurement_reports_the_collapsed_outcome_again():
    """Measuring the same qubit twice must not re-randomise it."""
    state = stabilizer_state(ghz_circuit(3, measure=False))
    first = state.measure(1)
    for _ in range(5):
        assert state.measure(1) is first
        # `outcome_of` is the probability of reading 1: certain, and equal to what
        # the first measurement returned.
        assert state.outcome_of(1) == float(first)
    # The state is the collapsed branch, not both branches: GHZ keeps all three bits
    # equal to the outcome that was just read.
    assert state.probability_of_bitstring((first, first, first)) == pytest.approx(1.0)
    assert state.probability_of_bitstring((not first, not first, not first)) == 0.0
    state.validate()


def test_a_y_basis_state_measures_randomly_and_each_branch_is_a_valid_state():
    """`|+i>` is a `Y` eigenstate and a 50/50 `Z` outcome; both branches must be clean."""
    seen = set()
    for seed in range(40):
        state = StabilizerState(1)
        state.apply_gate("h", (0,))      # |+>
        state.apply_gate("s", (0,))      # |+i>, a +1 eigenstate of Y
        assert state.expectation((2,)) == 1.0, "the state is not a Y eigenstate"
        assert state.outcome_of(0) == pytest.approx(0.5), "Z is undetermined"
        seen.add(state.measure(0, rng=np.random.default_rng(seed)))
        state.validate()
        branch = (state.probability_of_bitstring((True,)),
                  state.probability_of_bitstring((False,)))
        assert sorted(branch) == [0.0, 1.0], "a collapsed branch must be certain"
    assert seen == {True, False}


def test_y_measurement_direction_is_only_reachable_through_its_conjugation():
    """`Y` and `-Y` on a stabiliser state: one is deterministic, the other forbidden."""
    state = StabilizerState(1)          # +Z eigenstate
    state.apply_gate("h", (0,))         # +X eigenstate
    assert state.expectation((1,)) == 1.0
    assert state.expectation((3,)) == 0.0
    assert state.expectation((2,)) == 0.0
    state.apply_gate("s", (0,))         # +Y eigenstate
    assert state.expectation((2,)) == 1.0
    assert state.expectation((1,)) == 0.0
    assert state.expectation((3,)) == 0.0
    state.validate()


def test_a_zero_probability_branch_cannot_be_forced():
    state = StabilizerState(1)
    with pytest.raises(ValueError, match="zero probability"):
        state.measure(0, outcome=True)
    state.apply_gate("h", (0,))
    assert state.measure(0, outcome=True) is True


# ------------------------------------------------------------ measurement + feedforward
def test_teleportation_runs_identically_on_both_tiers():
    """`Measure`, conditioned gates and `Reset` exercise the whole op vocabulary."""
    ops = (
        Gate("h", (0,)),          # the message is |+>
        Gate("h", (1,)),          # Bell pair on (1, 2)
        Gate("cx", (1, 2)),
        Gate("cx", (0, 1)),       # Alice's Bell-basis measurement
        Gate("h", (0,)),
        Measure(0, 0),
        Measure(1, 1),
        Gate("x", (2,), condition=((1, True),)),
        Gate("z", (2,), condition=((0, True),)),
        Gate("h", (2,)),          # `|+>` measured in the X basis is always 0
        Measure(2, 2),
    )
    circuit = Circuit(ops=ops, num_qubits=3, num_bits=3)
    shots = 4000

    fast = StabilizerBackend().run(circuit, shots=shots, seed=21)
    dense = StatevectorBackend().run(circuit, shots=shots, seed=21)

    assert set(fast.counts) == set(dense.counts) == {"000", "010", "100", "110"}
    for count in fast.counts.values():
        assert count == pytest.approx(shots / 4, abs=4 * math.sqrt(shots * 0.1875) + 10)


def test_reset_returns_a_qubit_to_the_plus_one_state():
    deterministic = stabilizer_state(Circuit(ops=(Gate("h", (0,)),), num_qubits=1))
    deterministic.reset(0)
    assert deterministic.outcome_of(0) == 0.0
    assert deterministic.probability_of_bitstring((False,)) == 1.0

    # A reset of one half of an entangled pair is a random branch per trajectory --
    # the same thing the dense tier does per shot -- and the branch must be a
    # genuine post-measurement state, not a superposition of both outcomes.
    state = stabilizer_state(ghz_circuit(3, measure=False))
    state.reset(1)
    assert state.outcome_of(1) == 0.0
    branches = [state.probability_of_bitstring(bits)
                for bits in ((False, False, False), (True, False, True))]
    assert sorted(branches) == [0.0, 1.0]
    # `Z_0` and `Z_2` stay correlated in either branch: GHZ survives the reset of 1.
    assert state.expectation((3, 0, 3)) == 1.0


def test_the_distribution_limit_is_checked_before_anything_is_allocated():
    """A refusal that had already built the object would not be a limit."""
    import tracemalloc

    state = StabilizerState(20)
    tracemalloc.start()
    try:
        with pytest.raises(StabilizerUnsupportedError, match="anyone reads"):
            state.distribution()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # `2**20` outcomes would need at least a million tuples. The refusal costs the
    # message and the call frame, so a megabyte is a generous ceiling and still two
    # orders of magnitude below the object it refused to build.
    assert peak < 1_000_000


def test_an_exact_distribution_is_refused_past_the_enumerated_width():
    """`2**n` outcomes is a support, not a report: both engines refuse to build it.

    The stabilizer representation is compact, but the *distribution* is not: a state
    with `n` stabilizers can have up to `2**n` outcomes, so an exact enumeration is a
    request to allocate an exponential object. The Python tier and the Rust tier both
    refuse past twelve qubits rather than exhaust memory, and the refusal points at
    what does scale -- one outcome's probability, or sampling.
    """
    from apedp_dyn.vqc import native

    circuit = ghz_circuit(13, measure=False)
    state = StabilizerState(13)
    for op in circuit.ops:
        state.apply_gate(op.name, op.targets, op.params)
    with pytest.raises(StabilizerUnsupportedError, match="anyone reads"):
        state.distribution()
    native.reset_tallies()
    assert native.distribution(circuit) is None
    assert "distribution" in (native.last_failure() or "")
    assert native.tallies()["fallback"] == 1
    # The two things the refusal recommends still work, exactly.
    assert state.probability_of_bitstring((True,) * 13) == 0.5
    assert state.probability_of_bitstring((True,) + (False,) * 12) == 0.0
    # And the boundary is a boundary, not a guess: twelve qubits is still enumerated.
    assert len(stabilizer_state(ghz_circuit(12, measure=False)).distribution()) == 2


# --------------------------------------------- group membership, exhaustively
def _enumerated_group(state: StabilizerState) -> dict[tuple[int, ...], int]:
    """The group as ``{labels: phase}``, by multiplying the generators exactly.

    An independent path to the same answer: the string product is defined by the
    matrix table ``_PRODUCT`` and never touches the symplectic masks `phase_of`
    reduces over.
    """
    from apedp_dyn.vqc.stabilizer import I, _multiply

    members: dict[tuple[int, ...], int] = {}
    generators = list(zip(state.generators, state.phases, strict=True))
    for size in range(len(generators) + 1):
        for chosen in itertools.combinations(range(len(generators)), size):
            labels: tuple[int, ...] = tuple([I] * state.num_qubits)
            phase = 0
            for index in chosen:
                labels, phase = _multiply(labels, phase, *generators[index])
            members[labels] = phase % 4
    return members


@pytest.mark.parametrize("seed", [0, 2, 3, 8, 9, 17])
def test_group_membership_is_the_enumerated_group(seed):
    """Every Pauli string, not only the generators, and both answers are checked.

    ``phase_of`` is the whole of the state-level referee's decision, so a phase that
    depends on which rows the elimination happened to pick is a wrong verdict and not
    a rounding difference: it read ``+YYX`` as a stabilizer of a GHZ state it has zero
    overlap with, and failed to find ``Y_0 Y_1`` in the group of a Bell pair at all.
    """
    state = stabilizer_state(random_clifford_circuit(3, seed=seed))
    group = _enumerated_group(state)
    assert len(group) == 2 ** 3, "a pure state's group has one element per Pauli string"
    for labels in itertools.product(range(4), repeat=3):
        assert state.phase_of(labels) == group.get(labels), labels


def test_a_pauli_strings_phase_agrees_with_its_dense_expectation():
    """The same question asked of the dense tier, which shares no algebra with it.

    A signed Pauli string is in the stabilizer group exactly when the state is its
    ``+1`` or ``-1`` eigenstate, so the phase and ``<psi|P|psi>`` must agree: ``0``
    with ``+1``, ``2`` with ``-1``, and *absent* for every string whose expectation is
    zero. Odd phases are not stabilizers at all -- ``<psi|P|psi>`` is real -- and the
    referee must not report one.
    """
    circuit = random_clifford_circuit(3, seed=5, depth=16)
    dense = dense_state(circuit)
    state = stabilizer_state(circuit)
    for labels in itertools.product(range(4), repeat=3):
        expectation = dense_expectation(dense, labels)
        phase = state.phase_of(labels)
        if abs(expectation) < 1e-9:
            assert phase is None, (labels, phase, expectation)
            continue
        assert phase in (0, 2), (labels, phase, expectation)
        assert expectation == pytest.approx(1.0 if phase == 0 else -1.0, abs=1e-9)
