"""The phase-flip code and the five-qubit code, against their exact predictions.

Each code is tested on three things: it identifies the errors it is built for, it
fails on the errors it cannot handle, and its measured logical rate matches an
*independently computed* prediction. The predictions are arithmetic over the code
definition -- `3p^2 - 2p^3` for the repetition codes, an exhaustive 1024-pattern
enumeration for the five-qubit code -- not fits to the simulated data.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from apedp_dyn.quantum_compilation.statevector_sim import global_phase_free_equality
from apedp_dyn.vqc import StatevectorBackend, decode
from apedp_dyn.vqc.five_qubit import (
    DECODER,
    FIVE_QUBIT_STABILIZERS,
    READOUT_BITS,
    SYNDROME_BITS,
    encoded_state,
    exact_failure_probability,
    five_qubit_code_circuit,
    five_qubit_encoder_circuit,
    five_qubit_logical_error_rate,
    syndrome_of,
)
from apedp_dyn.vqc.logical import (
    analytic_failure_probability,
    logical_error_rate,
    phase_flip_code_circuit,
    single_error_syndrome,
)

SEED = 20261030


def sigma_band(probability: float, shots: int, sigma: float = 4.0) -> float:
    return sigma * np.sqrt(max(probability * (1 - probability), 1e-12) / shots)


# ------------------------------------------------------ phase-flip repetition code
@pytest.mark.parametrize("error_qubit", (0, 1, 2))
@pytest.mark.parametrize("prepare_logical_one", (False, True))
def test_the_phase_flip_code_identifies_and_corrects_every_single_z(error_qubit,
                                                                    prepare_logical_one):
    circuit = phase_flip_code_circuit(prepare_logical_one=prepare_logical_one,
                                      injected_errors=(error_qubit,))
    result = StatevectorBackend().run(circuit, shots=16, seed=SEED)
    syndrome = single_error_syndrome(error_qubit)
    readout = "1" * 3 if prepare_logical_one else "0" * 3
    key = "".join("1" if bit else "0" for bit in
                  (*syndrome, *([prepare_logical_one] * 3)))
    assert result.counts == {key: 16}
    assert decode(readout) is prepare_logical_one


def test_the_phase_flip_code_fails_on_two_errors():
    circuit = phase_flip_code_circuit(injected_errors=(0, 1))
    result = StatevectorBackend().run(circuit, shots=8, seed=SEED)
    key = next(iter(result.counts))
    assert key[:2] == "01", "the pair is misdiagnosed as a single error on qubit 2"
    assert decode(key[2:]) is True, "the logical value is flipped, not kept"


def test_the_phase_flip_rate_matches_3p_squared_minus_2p_cubed():
    probability, shots = 0.1, 4000
    run = logical_error_rate(probability, shots=shots, seed=SEED, code="phase_flip")
    assert run.code == "repetition-3-phaseflip"
    assert run.protects_against == "phase-flip only"
    assert run.rate == pytest.approx(analytic_failure_probability(probability),
                                     abs=sigma_band(0.028, shots))
    assert run.rate < run.uncorrected


def test_the_phase_flip_code_is_noiseless_at_zero_and_clean_at_full_noise():
    clean = logical_error_rate(0.0, shots=512, seed=SEED, code="phase_flip")
    assert clean.failures == 0
    # At p = 1 every qubit is flipped, so the code cannot help and reports ~1.
    saturated = logical_error_rate(1.0, shots=512, seed=SEED, code="phase_flip")
    assert saturated.rate > 0.6


# ------------------------------------------------------------- the five-qubit code
def test_the_code_is_perfect_fifteen_single_errors_sixteen_syndromes():
    assert len(FIVE_QUBIT_STABILIZERS) == 4
    assert len(DECODER) == 16
    assert DECODER[tuple([False] * 4)] == "IIIII"
    # All 15 single-qubit Paulis have distinct, non-zero syndromes.
    single = {syndrome_of("I" * q + p + "I" * (4 - q)) for q in range(5) for p in "XYZ"}
    assert len(single) == 15
    assert tuple([False] * 4) not in single


@pytest.mark.parametrize("prepare_logical_one", (False, True))
def test_the_encoder_prepares_the_exact_encoded_state_on_every_shot(prepare_logical_one):
    reference = encoded_state(prepare_logical_one)
    circuit = five_qubit_encoder_circuit(prepare_logical_one=prepare_logical_one)
    result = StatevectorBackend().run(circuit, shots=32, seed=SEED, keep_states=True)
    for state in result.final_states:
        # The four encoder ancillas must be back in |0>, so the data are the first
        # 32 amplitudes.
        assert global_phase_free_equality(state[:32], reference)
    assert all(excitation == 0.0 for excitation in result.ancilla_excitation.values())


@pytest.mark.parametrize("qubit", range(5))
@pytest.mark.parametrize("pauli", "XYZ")
@pytest.mark.parametrize("prepare_logical_one", (False, True))
def test_every_single_qubit_error_is_identified_and_corrected(qubit, pauli,
                                                             prepare_logical_one):
    circuit = five_qubit_code_circuit(prepare_logical_one=prepare_logical_one,
                                      injected_errors=((qubit, pauli),))
    sample = StatevectorBackend().run(circuit, shots=1, seed=SEED).samples[0]
    syndrome = tuple(sample[bit] for bit in SYNDROME_BITS)
    error = "I" * qubit + pauli + "I" * (4 - qubit)
    assert syndrome == syndrome_of(error)
    assert DECODER[syndrome] == error
    parity = bool(sum(bool(sample[bit]) for bit in READOUT_BITS) % 2)
    assert parity is prepare_logical_one, "the logical value must survive a single error"


def test_a_two_qubit_error_can_flip_the_logical_value():
    flips = 0
    for qubit_a, qubit_b in itertools.combinations(range(5), 2):
        circuit = five_qubit_code_circuit(injected_errors=((qubit_a, "X"), (qubit_b, "Z")))
        sample = StatevectorBackend().run(circuit, shots=1, seed=SEED).samples[0]
        if sum(bool(sample[bit]) for bit in READOUT_BITS) % 2:
            flips += 1
    assert flips > 0, "a weight-2 error must be able to escape a distance-3 code"


def test_the_exact_prediction_is_enumeration_not_a_fit():
    assert exact_failure_probability(0.0) == 0.0
    values = [exact_failure_probability(p) for p in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5)]
    assert values == sorted(values)
    # At p = 1/2 the error is spread over all 1024 patterns; the result is a
    # rational number with denominator 3^5, i.e. it came from summing weights.
    half = exact_failure_probability(0.5)
    assert 0.4 < half < 0.5
    assert round(half * 3 ** 5, 9) == round(half * 3 ** 5)
    assert exact_failure_probability(1.0) == pytest.approx(0.5432098765, abs=1e-6)


@pytest.mark.parametrize("probability", (0.05, 0.1))
def test_the_measured_rate_matches_the_exact_prediction(probability):
    shots = 2000
    run = five_qubit_logical_error_rate(probability, shots=shots, seed=SEED)
    assert run.rate == pytest.approx(run.exact, abs=sigma_band(run.exact, shots))
    assert run.rate < probability, "encoding must beat an unencoded physical error"
    payload = run.as_dict()
    assert payload["code"] == "five-qubit-perfect"
    assert payload["counts_phase_errors"] is False
