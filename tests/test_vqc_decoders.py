"""The decoder interface, the distance-3 table, and exact minimum-weight matching.

The matching decoder is checked against three independent things: the lookup table
it replaces (every distance-3 syndrome), a brute-force minimum-weight computation
written here in the test (every distance-7 syndrome), and the enumerated logical
failure prediction of the codes it drives.
"""
from __future__ import annotations

import itertools
import math

import pytest

from apedp_dyn.vqc.decoders import (
    Correction,
    Decoder,
    DecoderRefusalError,
    LookupDecoder,
    MatchingGraph,
    MinimumWeightDecoder,
    repetition_lookup_decoder,
    repetition_matching_graph,
    repetition_mwpm_decoder,
)
from apedp_dyn.vqc.logical import (
    analytic_failure_probability,
    decode,
    repetition_code_circuit,
    repetition_exact_failure_probability,
    repetition_logical_error_rate,
    single_error_syndrome,
)

SEED = 20261030


def error_string(distance: int, qubits, *, letter: str = "X") -> str:
    labels = ["I"] * distance
    for qubit in qubits:
        labels[qubit] = letter
    return "".join(labels)


def syndrome_of(error: str) -> tuple[bool, ...]:
    return tuple(error[check] != error[check + 1] for check in range(len(error) - 1))


def residual_logical_flip(error: str, correction: str) -> bool:
    residual = [first != second for first, second in zip(error, correction, strict=True)]
    return sum(residual) > len(residual) // 2


def brute_force_weight(graph: MatchingGraph, defects: list[int]) -> int:
    """Minimum total path weight over every pairing and boundary assignment."""
    reachable = [graph.shortest_paths(node) for node in defects]
    boundary = [min((table[b][0] for b in graph.boundaries if b in table), default=None)
                for table in reachable]

    def solve(indices: tuple[int, ...]) -> int:
        if not indices:
            return 0
        first, rest = indices[0], indices[1:]
        best = None
        if boundary[first] is not None:
            best = boundary[first] + solve(rest)
        for position, other in enumerate(rest):
            path = reachable[first].get(defects[other])
            if path is None:
                continue
            candidate = path[0] + solve(rest[:position] + rest[position + 1:])
            best = candidate if best is None else min(best, candidate)
        if best is None:
            raise AssertionError("unresolvable syndrome in the brute-force check")
        return best

    return solve(tuple(range(len(defects))))


# ---------------------------------------------------------------- the interface
def test_both_decoders_satisfy_the_interface():
    assert isinstance(repetition_lookup_decoder(3), Decoder)
    assert isinstance(repetition_mwpm_decoder(7), Decoder)
    assert repetition_lookup_decoder(3).distance == 3
    assert repetition_mwpm_decoder(7).name == "mwpm"


def test_the_lookup_table_is_the_one_the_distance_three_code_had():
    decoder = repetition_lookup_decoder(3)
    for qubit in range(3):
        correction = decoder.decode(single_error_syndrome(qubit))
        assert correction.pauli == error_string(3, (qubit,))
        assert correction.pauli_weight == 1
        assert correction.syndrome == single_error_syndrome(qubit)
    # The readout convention the table was built for: majority of the three bits.
    assert decode("111") and not decode("100") and not decode("010")


def test_mwpm_reproduces_the_lookup_answer_on_every_distance_three_syndrome():
    lookup, mwpm = repetition_lookup_decoder(3), repetition_mwpm_decoder(3)
    for pattern in range(1 << 2):
        syndrome = tuple(bool(pattern >> bit & 1) for bit in range(2))
        assert mwpm.decode(syndrome).pauli == lookup.decode(syndrome).pauli


def test_the_matching_is_minimum_not_greedy():
    """Every distance-7 syndrome, against a brute-force minimum written here."""
    decoder = repetition_mwpm_decoder(7)
    for pattern in range(1 << 6):
        syndrome = tuple(bool(pattern >> bit & 1) for bit in range(6))
        defects = [bit for bit, lit in enumerate(syndrome) if lit]
        if not defects:
            continue
        correction = decoder.decode(syndrome)
        expected = brute_force_weight(repetition_matching_graph(7), defects)
        assert correction.weight == expected, (syndrome, correction.pauli, expected)


def test_every_error_up_to_the_correction_capacity_is_decoded_exactly():
    for distance in (3, 5, 7):
        decoder = repetition_mwpm_decoder(distance)
        capacity = (distance - 1) // 2
        for weight in range(capacity + 1):
            for qubits in itertools.combinations(range(distance), weight):
                error = error_string(distance, qubits)
                correction = decoder.decode(syndrome_of(error))
                assert correction.pauli == error, (distance, error, correction.pauli)


def test_beyond_capacity_the_matching_can_miscorrect():
    decoder = repetition_mwpm_decoder(7)
    miscorrected = 0
    for qubits in itertools.combinations(range(7), 4):
        error = error_string(7, qubits)
        correction = decoder.decode(syndrome_of(error))
        if correction.pauli != error:
            miscorrected += 1
    assert miscorrected > 0, "a distance-7 code must be defeatable by a weight-4 error"


def test_the_refusals_are_named():
    with pytest.raises(DecoderRefusalError, match="distance 3"):
        repetition_lookup_decoder(5)
    with pytest.raises(ValueError, match="odd"):
        repetition_matching_graph(4)
    with pytest.raises(ValueError, match="odd"):
        repetition_matching_graph(1)
    mwpm = repetition_mwpm_decoder(5)
    with pytest.raises(DecoderRefusalError, match="expected 4 syndrome bits"):
        mwpm.decode((True,))
    lookup = repetition_lookup_decoder(3)
    with pytest.raises(DecoderRefusalError, match="not in the lookup table"):
        lookup.decode((False,))  # a distance-5 syndrome given to a distance-3 table


def test_an_isolated_defect_is_refused_rather_than_guessed():
    graph = MatchingGraph(2)          # two syndrome nodes, no boundary, one edge
    graph.add_edge(0, 1, 1, "XI")
    decoder = MinimumWeightDecoder(graph, distance=2)
    assert decoder.decode((True, True)).pauli == "XI"
    with pytest.raises(DecoderRefusalError, match="isolated"):
        decoder.decode((True, False))


def test_the_graph_composes_paths_and_routes_to_boundaries():
    graph = MatchingGraph(2)
    graph.add_boundary(2)
    graph.add_edge(0, 1, 1, "XI")
    graph.add_edge(1, 2, 1, "IX")
    decoder = MinimumWeightDecoder(graph, distance=2)
    assert decoder.decode((True, True)).pauli == "XI"
    correction = decoder.decode((False, True))
    assert correction.pauli == "IX" and correction.weight == 1
    assert graph.shortest_paths(0)[2][1] == "XX"


# ------------------------------------------------------ driving the code circuits
@pytest.mark.parametrize("distance", (3, 5, 7))
def test_the_measured_rate_matches_the_enumerated_prediction(distance: int):
    probability, shots = 0.1, 1200
    run = repetition_logical_error_rate(distance, probability, shots=shots, seed=SEED)
    band = 4.0 * math.sqrt(max(run.exact * (1 - run.exact), 1e-12) / shots)
    assert run.decoder_name == "mwpm"
    assert run.rate == pytest.approx(run.exact, abs=band)
    assert run.rate < run.uncorrected


def test_the_prediction_is_the_analytic_one_at_distance_three():
    for probability in (0.02, 0.05, 0.1, 0.2):
        assert repetition_exact_failure_probability(
            3, probability) == pytest.approx(analytic_failure_probability(probability),
                                             abs=1e-15)


def test_more_distance_is_less_logical_error_at_the_same_physics():
    values = [repetition_exact_failure_probability(d, 0.1) for d in (3, 5, 7)]
    assert values == sorted(values, reverse=True)
    assert values[-1] < values[0] / 5


def test_the_decoder_is_an_input_to_the_circuit():
    lookup_circuit = repetition_code_circuit(3, error_probability=0.1, decoder="lookup")
    mwpm_circuit = repetition_code_circuit(3, error_probability=0.1, decoder="mwpm")
    assert lookup_circuit.metadata["decoder"] == "lookup"
    assert mwpm_circuit.metadata["decoder"] == "mwpm"
    assert lookup_circuit.sha256() != mwpm_circuit.sha256()

    # The corrections agree at distance 3, so the two circuits must produce the same
    # counts at the same seed.
    from apedp_dyn.vqc import StatevectorBackend

    first = StatevectorBackend().run(lookup_circuit, shots=200, seed=SEED)
    second = StatevectorBackend().run(mwpm_circuit, shots=200, seed=SEED)
    assert first.counts == second.counts


def test_the_two_error_types_are_the_same_experiment_in_a_rotated_basis():
    bit_flip = repetition_logical_error_rate(7, 0.1, shots=400, seed=SEED, code="bit_flip")
    phase_flip = repetition_logical_error_rate(7, 0.1, shots=400, seed=SEED,
                                               code="phase_flip")
    assert bit_flip.rate == phase_flip.rate
    assert bit_flip.exact == phase_flip.exact
    assert phase_flip.as_dict()["code"] == "repetition-7-phase_flip"


def test_the_correction_object_carries_its_evidence():
    correction = repetition_mwpm_decoder(7).decode((True, False, False, False, False, True))
    assert isinstance(correction, Correction)
    # Defects at the two ends are cheaper to route to the boundaries (1 + 1) than to
    # pair with each other along the chain (5), which is the matching doing its job.
    payload = correction.as_dict()
    assert payload["pauli"] == "XIIIIIX"
    assert payload["support"] == [0, 6]
    assert payload["matching_weight"] == 2
    assert payload["decoder"] == "mwpm"


def test_the_lookup_decoder_validates_its_table():
    with pytest.raises(DecoderRefusalError, match="data qubits"):
        LookupDecoder({(False, False): "XI"}, distance=3)
    with pytest.raises(DecoderRefusalError, match="Pauli"):
        LookupDecoder({(False, False): "ABC"}, distance=3)
