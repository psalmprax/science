"""APEDP-VQC Core acceptance tests.

Seven criteria, each measured rather than asserted:

1. a common, versioned circuit IR that serialises, hashes and round-trips without
   changing meaning;
2. Bell, GHZ and teleportation executed by an exact statevector backend, compared
   against independently computed states and distributions;
3. measurement, classical bits and feedforward actually driving the circuit
   (teleportation does not work without them);
4. density-matrix noise channels that preserve trace and positivity and match
   analytic answers;
5. one working stochastic noise channel in the circuit IR, reproducible from its
   seed, showing the expected physical trend;
6. a complete three-qubit bit-flip correction experiment: every single error
   corrected, two errors not, logical rate matching `3p^2 - 2p^3`;
7. machine-readable receipts that say VALID, INVALID or UNRESOLVED, and reject
   deliberately wrong candidates.

The tests are deliberately about *physics* rather than structure: a class named
`Simulator` is not evidence, a state that matches (|00> + |11>)/sqrt(2) is.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from apedp_dyn.quantum_compilation.statevector_sim import global_phase_free_equality
from apedp_dyn.vqc import (
    Circuit,
    CircuitValidationError,
    DensityMatrix,
    Gate,
    Measure,
    Noise,
    Reset,
    StatevectorBackend,
    adjudicate,
    bit_flip_code_circuit,
    bitstring,
    decode,
    depolarizing,
    logical_error_rate,
    purity_of_bell_after_depolarizing,
    single_error_syndrome,
)
from apedp_dyn.vqc.noise import CHANNELS, amplitude_damping, bit_flip, phase_flip

ROOT_SEED = 20261030


# --------------------------------------------------------------------- helpers
def sigma_band(probability: float, shots: int, sigma: float = 4.0) -> float:
    return sigma * math.sqrt(max(probability * (1.0 - probability), 1e-12) / shots)


def teleportation_circuit(*, z_correction: bool = True, readout: bool = True,
                          dirty_ancilla: bool = False) -> Circuit:
    """Teleport `RY(pi/4)|0>` from qubit 0 to qubit 2.

    Qubit 1 is the Bell-pair half that is consumed and must return to |0>; it is
    declared as an ancilla so the receipt checks the cleanup. The final Hadamard
    and measurement read qubit 2 in the X basis, where the prepared state leaves a
    signature (`P(0) = (1 + cos pi/4)/2`) that a missing phase correction destroys.
    """
    ops = [
        Gate("ry", (0,), (math.pi / 4.0,)),
        Gate("h", (1,)),
        Gate("cx", (1, 2)),
        Gate("cx", (0, 1)),
        Gate("h", (0,)),
        Measure(0, 0),
        Measure(1, 1),
        Gate("x", (2,), condition=((1, True),)),
    ]
    if z_correction:
        ops.append(Gate("z", (2,), condition=((0, True),)))
    if dirty_ancilla:
        ops.append(Gate("x", (1,)))  # leaves garbage on the consumed qubit
    if readout:
        ops += [Gate("h", (2,)), Measure(2, 2)]
    return Circuit(ops=tuple(ops), num_qubits=3,
                   num_bits=3 if readout else 2, ancillas=(1,),
                   metadata={"generator": "tests.teleportation"})


def teleported_x_basis_expectation() -> dict[str, float]:
    """The exact readout distribution: the two teleport bits are uniform, q2 is RY(pi/4)."""
    p_zero = (1.0 + math.cos(math.pi / 4.0)) / 2.0
    expected = {}
    for first in ("0", "1"):
        for second in ("0", "1"):
            expected[first + second + "0"] = 0.25 * p_zero
            expected[first + second + "1"] = 0.25 * (1.0 - p_zero)
    return expected


# ------------------------------------------------------- 1. the representation
def test_the_same_circuit_serialises_round_trips_and_hashes_the_same():
    circuit = teleportation_circuit()
    text = circuit.to_json()

    rebuilt = Circuit.from_json(text)
    assert rebuilt == circuit
    assert rebuilt.sha256() == circuit.sha256()
    assert rebuilt.to_json() == text, "serialisation must be canonical, not merely stable per object"

    # Canonical: no whitespace, keys sorted, identical for an equal circuit built fresh.
    assert " " not in text
    assert Circuit.from_json(text).to_json() == text

    # Executing the original and the reconstruction must agree shot for shot.
    original = StatevectorBackend().run(circuit, shots=512, seed=ROOT_SEED)
    reconstructed = StatevectorBackend().run(rebuilt, shots=512, seed=ROOT_SEED)
    assert original.counts == reconstructed.counts
    assert original.samples == reconstructed.samples


def test_the_hash_is_an_identity_not_a_formality():
    circuit = teleportation_circuit()
    different_gate = teleportation_circuit(z_correction=False)
    relabelled = circuit.with_metadata(label="other")
    assert circuit.sha256() != different_gate.sha256()
    assert circuit.sha256() != relabelled.sha256()
    assert len(circuit.sha256()) == 64


@pytest.mark.parametrize("ops,num_qubits,num_bits,kwargs,reason", [
    ((Gate("nope", (0,)),), 1, 0, {}, "unknown gate"),
    ((Gate("cx", (0,)),), 2, 0, {}, "wrong arity"),
    ((Gate("cx", (0, 0)),), 1, 0, {}, "repeated qubit"),
    ((Gate("x", (3,)),), 2, 0, {}, "qubit out of range"),
    ((Gate("x", (0,), condition=((0, True),)),), 1, 1, {}, "condition read before written"),
    ((Measure(0, 2),), 1, 1, {}, "measurement bit out of range"),
    ((Noise(0, "bit_flip", 1.5),), 1, 0, {}, "probability out of range"),
    ((Noise(0, "bit_flip", 0.1), Noise(0, "cosmic_ray", 0.1)), 1, 0, {}, "unknown noise kind"),
    ((Gate("rx", (0,)),), 1, 0, {}, "missing parameter"),
    ((), 0, 0, {}, "no qubits"),
    ((), 2, 0, {"ancillas": (2,)}, "ancilla out of range"),
    ((), 2, 0, {"schema_version": 99}, "unknown schema version"),
])
def test_the_ir_refuses_programs_a_backend_would_have_to_guess_about(
        ops, num_qubits, num_bits, kwargs, reason):
    circuit = Circuit(ops=tuple(ops), num_qubits=num_qubits, num_bits=num_bits, **kwargs)
    with pytest.raises(CircuitValidationError):
        circuit.validate()


# ------------------------------------------------------- 2. exact simulation
def test_bell_state_is_the_entangled_state_and_its_marginals_are_uniform():
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1))), num_qubits=2)
    result = StatevectorBackend().run(circuit, shots=1, seed=1, keep_states=True)
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = expected[3] = 1.0 / math.sqrt(2.0)
    assert global_phase_free_equality(result.final_states[0], expected)

    # Measured, it is perfectly correlated and each outcome is equiprobable.
    measured = Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                            Measure(0, 0), Measure(1, 1)), num_qubits=2, num_bits=2)
    shots = 4096
    counts = StatevectorBackend().run(measured, shots=shots, seed=ROOT_SEED).counts
    assert set(counts) == {"00", "11"}
    assert counts["00"] / shots == pytest.approx(0.5, abs=sigma_band(0.5, shots))


def test_ghz_state_entangles_three_qubits():
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)), Gate("cx", (1, 2)),
                           Measure(0, 0), Measure(1, 1), Measure(2, 2)),
                      num_qubits=3, num_bits=3)
    shots = 4096
    counts = StatevectorBackend().run(circuit, shots=shots, seed=ROOT_SEED).counts
    assert set(counts) == {"000", "111"}
    assert counts["111"] / shots == pytest.approx(0.5, abs=sigma_band(0.5, shots))


def test_teleportation_moves_the_state_exactly_on_every_shot():
    """Not just the statistics: after the consumed qubits are reset, the state *is* |psi>."""
    circuit = teleportation_circuit(readout=False)
    result = StatevectorBackend().run(circuit, shots=64, seed=ROOT_SEED, keep_states=True)
    theta = math.pi / 4.0
    psi = np.array([math.cos(theta / 2.0), math.sin(theta / 2.0)], dtype=np.complex128)
    target = np.zeros(8, dtype=np.complex128)
    target[0] = psi[0]       # q0 = q1 = |0>
    target[4] = psi[1]       # q2 carries psi
    assert all(global_phase_free_equality(state, target) for state in result.final_states)
    assert result.ancilla_excitation[1] == pytest.approx(0.0, abs=1e-12)


def test_teleportation_needs_the_classical_corrections():
    """Without feedforward or the phase fixup, the state does not arrive."""
    shots = 4096
    expected = teleported_x_basis_expectation()
    good = StatevectorBackend().run(teleportation_circuit(), shots=shots, seed=ROOT_SEED)
    p_zero = sum(count for key, count in good.counts.items() if key.endswith("0")) / shots
    assert p_zero == pytest.approx((1.0 + math.cos(math.pi / 4.0)) / 2.0,
                                   abs=sigma_band(0.85, shots))

    # Dropping the conditional Z leaves the phase correction to chance: half the
    # branches are Z-flipped, so the X-basis odds collapse to a coin flip.
    partial = teleportation_circuit(z_correction=False)
    without_phase = StatevectorBackend().run(partial, shots=shots, seed=ROOT_SEED)
    p_zero_partial = sum(c for k, c in without_phase.counts.items() if k.endswith("0")) / shots
    assert p_zero_partial == pytest.approx(0.5, abs=sigma_band(0.5, shots))
    assert abs(p_zero_partial - (1.0 + math.cos(math.pi / 4.0)) / 2.0) > 0.3

    # And the declared expectation is the one the correct circuit meets.
    assert set(good.counts) == set(expected)


# ------------------------------------------------------------- 3. noise, exact
CHANNEL_PROBABILITIES = (0.0, 0.05, 0.2, 0.5, 1.0)


@pytest.mark.parametrize("name", sorted(CHANNELS))
@pytest.mark.parametrize("probability", CHANNEL_PROBABILITIES)
def test_every_channel_preserves_a_valid_density_matrix(name, probability):
    state = DensityMatrix(1)
    state.set_pure(np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0))
    state.apply_channel(0, name, probability)
    assert state.trace() == pytest.approx(1.0, abs=1e-12)
    assert state.is_valid()
    assert state.purity() <= 1.0 + 1e-12


def test_depolarizing_a_bell_half_matches_the_analytic_fidelity_and_purity():
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    for probability in CHANNEL_PROBABILITIES:
        state = DensityMatrix(2)
        state.set_pure(bell)
        state.apply_channel(0, "depolarizing", probability)
        # Half of a Bell pair depolarised into the Pauli basis is a uniform mixture
        # over the four Bell states: fidelity 1-p, purity (1-p)^2 + p^2/3.
        assert state.fidelity_with(bell) == pytest.approx(1.0 - probability, abs=1e-12)
        assert state.purity() == pytest.approx(
            purity_of_bell_after_depolarizing(probability), abs=1e-12)
        assert state.is_valid()


def test_the_other_channels_match_their_definitions():
    damped = DensityMatrix(1)
    damped.set_basis_state(1)
    damped.apply_channel(0, "amplitude_damping", 0.4)
    assert damped.probabilities(0) == pytest.approx((0.4, 0.6), abs=1e-12)

    flipped = DensityMatrix(1)
    flipped.apply_channel(0, "bit_flip", 0.3)
    assert flipped.probabilities(0) == pytest.approx((0.7, 0.3), abs=1e-12)

    dephased = DensityMatrix(1)
    dephased.set_pure(np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0))
    dephased.apply_channel(0, "dephasing", 0.5)
    assert dephased.purity() == pytest.approx(0.75, abs=1e-12)

    # Kraus completeness, checked rather than assumed: a valid channel's operators
    # satisfy sum_k K_k^dagger K_k = I, which is what makes it trace preserving.
    for name, channel in CHANNELS.items():
        kraus = channel(0.3)
        completeness = sum(operator.conj().T @ operator for operator in kraus)
        assert np.allclose(completeness, np.eye(2)), name
    assert len(bit_flip(0.5)) == len(phase_flip(0.5)) == 2
    assert len(depolarizing(0.5)) == 4
    assert len(amplitude_damping(0.5)) == 2


# --------------------------------------- 4. noise in the IR, sampled with a seed
def test_noise_trajectories_are_reproducible_and_show_the_physical_trend():
    shots = 8000
    probability = 0.2
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                           Noise(0, "bit_flip", probability),
                           Measure(0, 0), Measure(1, 1)), num_qubits=2, num_bits=2)
    first = StatevectorBackend().run(circuit, shots=shots, seed=ROOT_SEED)
    second = StatevectorBackend().run(circuit, shots=shots, seed=ROOT_SEED)
    third = StatevectorBackend().run(circuit, shots=shots, seed=ROOT_SEED + 1)

    assert first.counts == second.counts, "a seed must reproduce the run exactly"
    assert first.counts != third.counts

    # A bit flip on one half of a Bell pair decorrelates the pair with probability p.
    same = first.counts.get("00", 0) + first.counts.get("11", 0)
    assert same / shots == pytest.approx(1.0 - probability, abs=sigma_band(0.8, shots))
    assert 0.7 < same / shots < 0.9


def test_noise_free_runs_are_deterministic_across_backend_instances():
    circuit = bit_flip_code_circuit(error_probability=0.0)
    first = StatevectorBackend().run(circuit, shots=256, seed=7)
    second = StatevectorBackend().run(circuit, shots=256, seed=7)
    assert first.counts == second.counts == {"00000": 256}


# ------------------------------------------------------- 5. the logical qubit
@pytest.mark.parametrize("error_qubit", (0, 1, 2))
@pytest.mark.parametrize("prepare_logical_one", (False, True))
def test_the_code_identifies_and_corrects_every_single_error(error_qubit, prepare_logical_one):
    circuit = bit_flip_code_circuit(prepare_logical_one=prepare_logical_one,
                                    injected_errors=(error_qubit,))
    result = StatevectorBackend().run(circuit, shots=16, seed=ROOT_SEED)
    syndrome = single_error_syndrome(error_qubit)
    expected_key = bitstring((*syndrome, *([prepare_logical_one] * 3)))
    assert result.counts == {expected_key: 16}, (
        f"X on qubit {error_qubit} should give syndrome {syndrome} and a corrected readout")


def test_two_errors_are_miscorrected_into_a_logical_failure():
    """Two bit flips give syndrome (0, 1), so the decoder "fixes" qubit 2 and completes
    the triple: the readout is 111 and the logical value has flipped."""
    circuit = bit_flip_code_circuit(injected_errors=(0, 1))
    result = StatevectorBackend().run(circuit, shots=8, seed=ROOT_SEED)
    key = next(iter(result.counts))
    assert key[:2] == "01", "the pair is misdiagnosed as a single error on qubit 2"
    assert key[2:] == "111"
    assert decode(key[2:]) is True, "the logical value is flipped, not kept"


def test_reset_returns_a_qubit_to_zero_whatever_it_held():
    for preparation in (Gate("x", (0,)), Gate("h", (0,))):
        circuit = Circuit(ops=(preparation, Reset(0), Measure(0, 0)),
                          num_qubits=1, num_bits=1)
        result = StatevectorBackend().run(circuit, shots=128, seed=ROOT_SEED)
        assert result.counts == {"0": 128}
        assert circuit.resources()["resets"] == 1
        assert Circuit.from_json(circuit.to_json()) == circuit


def test_reset_is_trace_preserving_on_an_entangled_pair():
    """Resetting one half of a Bell pair leaves the other maximally mixed.

    A projection that quietly dropped the state would leave the partner in |0>;
    the reset map cuts the correlation and leaves the reduced state intact.
    """
    circuit = Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)), Reset(0),
                           Measure(1, 0)), num_qubits=2, num_bits=1)
    shots = 4096
    counts = StatevectorBackend().run(circuit, shots=shots, seed=ROOT_SEED).counts
    assert counts.get("1", 0) / shots == pytest.approx(0.5, abs=sigma_band(0.5, shots))


def test_the_logical_rate_matches_its_exact_prediction_and_beats_the_physical_rate():
    probability = 0.1
    shots = 4000
    run = logical_error_rate(probability, shots=shots, seed=ROOT_SEED)
    assert run.failures > 0
    assert run.rate == pytest.approx(run.analytic, abs=sigma_band(run.analytic, shots))
    assert run.analytic == pytest.approx(3 * probability ** 2 - 2 * probability ** 3)
    # The point of encoding: below the physical rate at a noise level the code covers.
    assert run.rate < run.uncorrected


def test_a_noiseless_code_never_fails():
    for prepared in (False, True):
        run = logical_error_rate(0.0, shots=512, seed=ROOT_SEED,
                                 prepare_logical_one=prepared)
        assert run.failures == 0
        assert run.analytic == 0.0


# ------------------------------------------------------------ 6. receipts
def test_the_correct_teleportation_earns_a_valid_receipt():
    receipt = adjudicate(teleportation_circuit(), expected=teleported_x_basis_expectation(),
                         shots=4096, seed=ROOT_SEED, min_effect=0.1,
                         provenance={"generator": "tests.teleportation", "search": "none"})
    assert receipt.verdict == "VALID", receipt.failed()
    assert receipt.failed() == ()

    payload = json.loads(receipt.to_json())
    assert payload["verdict"] == "VALID"
    assert payload["identity"]["circuit_sha256"] == teleportation_circuit().sha256()
    assert payload["resources"]["qubits"] == 3
    assert payload["resources"]["ancillas"] == 1
    assert payload["ancillas"]["worst_excitation"] == pytest.approx(0.0, abs=1e-12)
    assert payload["reproducibility"]["seed"] == ROOT_SEED
    assert payload["statistics"]["required_shots"] >= 1


def test_the_receipt_rejects_a_circuit_that_changes_behaviour_on_a_superposition():
    receipt = adjudicate(teleportation_circuit(z_correction=False),
                         expected=teleported_x_basis_expectation(),
                         shots=4096, seed=ROOT_SEED, min_effect=0.1)
    assert receipt.verdict == "INVALID"
    assert "outcome distribution matches the declared expectation" in receipt.failed()
    worst = next(check for check in receipt.checks
                 if check.name.startswith("outcome distribution"))
    assert abs(worst.measured["worst_z"]) > 10


def test_the_receipt_rejects_garbage_left_in_an_ancilla():
    receipt = adjudicate(teleportation_circuit(dirty_ancilla=True),
                         expected=teleported_x_basis_expectation(),
                         shots=4096, seed=ROOT_SEED, min_effect=0.1)
    assert receipt.verdict == "INVALID"
    assert "declared ancillas return to |0>" in receipt.failed()
    assert receipt.ancillas["worst_excitation"] > 0.99


def test_the_receipt_abstains_rather_than_passing_an_underpowered_run():
    receipt = adjudicate(teleportation_circuit(), expected=teleported_x_basis_expectation(),
                         shots=8, seed=ROOT_SEED, min_effect=0.1)
    assert receipt.verdict == "UNRESOLVED"
    assert receipt.failed() == ()
    assert "the test had power to detect the declared effect" in receipt.unresolved()


def test_an_invalid_circuit_gets_an_invalid_receipt_rather_than_an_exception():
    circuit = Circuit(ops=(Gate("nope", (0,)),), num_qubits=1)
    receipt = adjudicate(circuit, expected={"0": 1.0}, shots=16, seed=ROOT_SEED)
    assert receipt.verdict == "INVALID"
    assert "circuit is a valid VQC program" in receipt.failed()
    assert json.loads(receipt.to_json())["verdict"] == "INVALID"
