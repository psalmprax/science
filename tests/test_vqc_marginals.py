"""Exact marginals as the sampling referee above the enumerable width.

The sampling comparison was refereed by the exact full distribution, which is enumerable
only up to 12 qubits. These pins hold the replacement to three claims:

* **the marginal is the specification's, not the sampler's.** Every marginal here is
  checked against a dense partial trace of a state built by projecting ``|0...0>`` onto
  the specification's eigenspaces -- an independent route through matrix algebra rather
  than through the exponent tricks the module uses. A wrong marginal would be a wrong
  referee, and a silent one is worse than none;
* **the verdict is a test, not a number.** The chi-square survival function is pinned
  against textbook critical values, the Bonferroni correction is pinned as arithmetic
  (`min(1, m * p)`), and a subset whose expected counts are too small abstains as
  `UNRESOLVED` instead of quoting a p-value nobody should trust;
* **the blind spot is measured.** A graph state is uniform in the Z basis, so no Z-basis
  marginal separates one graph state from another: samples of a *different* graph state
  pass this referee, and that is pinned as a fact about the referee rather than left as
  a sentence in a docstring.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from apedp_dyn.vqc.corpus import random_stabilizer_target
from apedp_dyn.vqc.external import ToolUnavailable, apedp_sampling, stim_sampling
from apedp_dyn.vqc.marginals import (
    FAMILY_ALPHA,
    MIN_EXPECTED_COUNT,
    MarginalUnsupported,
    adjudicate_marginal,
    chi_square_sf,
    exact_marginal,
    marginal_counts,
    marginal_pair,
)
from apedp_dyn.vqc.synthesis import (
    cluster_state_target,
    ghz_target,
    ring_graph_target,
    seeded_graph_target,
)

PAULI = {
    0: np.eye(2, dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),                          # X
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),                       # Y
    3: np.array([[1, 0], [0, -1]], dtype=complex),                         # Z
}


def dense_state(target) -> np.ndarray:
    """``|psi>`` by projecting a full-support start onto the stabilizer eigenspaces.

    An independent oracle: real Pauli matrices and matrix products, so a Y generator
    contributes its actual ``i`` rather than a bit rule. The start has to have support on
    the target -- ``|0...0>`` does not for a state like ``|11>`` -- so it is a fixed
    seeded full-support vector, re-drawn if the projections annihilate it.
    """
    n = target.num_qubits
    for seed in range(8):
        rng = np.random.default_rng(seed)
        state = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        state /= np.linalg.norm(state)
        for labels, phase in target.stabilizers:
            matrix = np.eye(1, dtype=complex)
            for label in labels:
                matrix = np.kron(matrix, PAULI[int(label)])
            sign = 1.0 if int(phase) % 4 == 0 else -1.0
            state = 0.5 * (state + sign * (matrix @ state))
            norm = np.linalg.norm(state)
            if norm < 1e-9:
                break
            state = state / norm
        else:
            return np.abs(state) ** 2
    raise AssertionError("the projections annihilated every start vector")


def dense_marginal(probabilities: np.ndarray, num_qubits: int, subset) -> dict:
    """The partial trace of a dense probability vector, over one subset of wires."""
    out: dict[tuple[bool, ...], float] = {}
    for index in range(1 << num_qubits):
        # `specification`-ordered: qubit 0 is the most significant bit.
        bits = tuple(bool((index >> (num_qubits - 1 - qubit)) & 1)
                     for qubit in range(num_qubits))
        key = tuple(bits[qubit] for qubit in subset)
        out[key] = out.get(key, 0.0) + float(probabilities[index])
    return out


def _subsets(num_qubits: int) -> list[tuple[int, ...]]:
    stride = max(1, num_qubits // 4)
    return [(0,), (num_qubits - 1,), (0, 1), (0, 2),
            tuple(sorted(range(0, num_qubits, stride)))[:4],
            tuple(range(min(3, num_qubits)))]


# ------------------------------------------------------------------ the marginal
@pytest.mark.parametrize("target", [
    ghz_target(6),
    cluster_state_target(2, 3),
    ring_graph_target(6),
    seeded_graph_target(6, extra_edge_probability=0.4, seed=11),
    random_stabilizer_target(6, seed=51_000),
    random_stabilizer_target(7, seed=51_007),
])
def test_the_marginal_is_the_specifications_dense_partial_trace(target):
    probabilities = dense_state(target)
    for subset in _subsets(target.num_qubits):
        marginal = exact_marginal(target, subset)
        expected = dense_marginal(probabilities, target.num_qubits, subset)
        support = {bits: p for bits, p in expected.items() if p > 1e-12}
        assert set(marginal.probabilities) == set(support), (target.name, subset)
        for bits, probability in support.items():
            assert marginal.probabilities[bits] == pytest.approx(probability, abs=1e-12)
        assert sum(marginal.probabilities.values()) == pytest.approx(1.0, abs=1e-12)


def test_a_sign_is_honoured_not_assumed():
    """The generators' signs decide the distribution: ``-Z`` is not ``+Z``.

    A referee that read only the Pauli labels and dropped the sign would call both of
    these ``|00>``, and every sample of ``|11>`` would be reported as inconsistent. The
    dense oracle agrees, so this is not the module checking itself.
    """
    from apedp_dyn.vqc.synthesis import I, TargetSpecification, Z

    ones = TargetSpecification(
        "one-one", 2, (((Z, I), 2), ((I, Z), 2)))
    zeros = TargetSpecification(
        "zero-zero", 2, (((Z, I), 0), ((I, Z), 0)))
    for target, bit in ((ones, True), (zeros, False)):
        marginal = exact_marginal(target, (0, 1))
        assert set(marginal.probabilities) == {(bit, bit)}
        assert marginal.probability == pytest.approx(1.0)
        assert dense_state(target)[0b11 if bit else 0b00] == pytest.approx(
            1.0, abs=1e-12)


def test_a_graph_state_marginal_is_uniform_on_every_subset():
    """The referee's blind spot, as arithmetic rather than as prose."""
    target = ring_graph_target(10)
    for size in (1, 2, 3, 5):
        marginal = exact_marginal(target, tuple(range(size)))
        assert marginal.is_uniform, size
        assert marginal.support == marginal.cells == 2 ** size
        assert marginal.probability == pytest.approx(2.0 ** -size)


def test_a_ghz_marginal_keeps_its_correlation():
    target = ghz_target(16)
    marginal = exact_marginal(target, (3, 7, 11))
    assert set(marginal.probabilities) == {(False, False, False), (True, True, True)}
    assert marginal.probability == pytest.approx(0.5)
    assert marginal.dimension == 15


def test_the_marginal_is_cheap_where_the_state_is_wide():
    """Linear algebra, not enumeration: a 32-qubit state's marginal is instant.

    The pin is a cost bound, not a timing claim: the algorithm is Gaussian elimination
    over GF(2) plus ``2**|T|`` outcomes, so the 32-qubit case must not be measurably
    worse than the 12-qubit one. A regression to enumeration would blow past this.
    """
    target = random_stabilizer_target(32, seed=50_320)
    start = time.perf_counter()
    marginal = exact_marginal(target, tuple(range(10)))
    seconds = time.perf_counter() - start
    assert marginal.cells == 1_024
    assert 0 < marginal.support <= marginal.cells
    assert seconds < 2.0, seconds


def test_a_subset_that_is_not_a_subset_is_refused():
    target = ghz_target(6)
    for subset in ((), (0, 0), (0, 6), (-1, 2)):
        with pytest.raises(MarginalUnsupported):
            exact_marginal(target, subset)
    with pytest.raises(MarginalUnsupported):
        exact_marginal(target, tuple(range(13)))
    with pytest.raises(MarginalUnsupported):
        marginal_counts({(False, True): 3}, (2,))


# ----------------------------------------------------------------- the statistics
@pytest.mark.parametrize("statistic, dof, expected", [
    (3.841458820694124, 1, 0.05),
    (5.991464547107979, 2, 0.05),
    (7.814727903251179, 3, 0.05),
    (11.344866730144373, 3, 0.01),
    (23.209251158954356, 10, 0.01),
])
def test_the_chi_square_survival_matches_textbook_critical_values(statistic, dof, expected):
    assert chi_square_sf(statistic, dof) == pytest.approx(expected, abs=1e-9)


def test_the_chi_square_survival_is_monotone_and_well_behaved():
    previous = 1.0
    for statistic in (0.0, 0.5, 1.0, 5.0, 50.0, 500.0, 5_000.0):
        value = chi_square_sf(statistic, 1_023)
        assert 0.0 <= value <= previous
        previous = value
    assert chi_square_sf(1_022.0, 1_023) == pytest.approx(0.5, abs=5e-3)
    with pytest.raises(ValueError):
        chi_square_sf(1.0, 0)


# ------------------------------------------------------------- one sample set
@pytest.fixture(scope="module")
def ghz_32():
    return ghz_target(32)


def _wide_subsets(num_qubits: int) -> list[tuple[int, ...]]:
    return ([(wire, wire + 1) for wire in range(num_qubits - 1)]
            + [tuple(range(6)), tuple(range(num_qubits - 6, num_qubits))])


def test_a_correct_sampler_is_consistent_above_the_enumerable_width(ghz_32):
    subsets = _wide_subsets(ghz_32.num_qubits)
    samples = apedp_sampling(ghz_32, shots=4_000, seed=5)
    block = adjudicate_marginal(samples, ghz_32, subsets)
    assert block["qubits"] == 32
    assert block["verdict"] == "CONSISTENT"
    assert block["family_size"] == len(subsets)
    assert block["resolved_subsets"] == len(subsets)
    assert block["unresolved_subsets"] == []
    for row in block["subsets"]:
        assert row["verdict"] == "CONSISTENT"
        assert row["family_p_value"] == min(1.0, row["p_value"] * len(subsets))
        assert row["dof"] == row["support"] - 1
        assert row["outcomes_outside_the_exact_support"] == 0
        assert row["target_sha256"] == ghz_32.sha256()


def test_a_miswired_wire_is_caught_and_named(ghz_32):
    """A wrong wire is found by the subsets that contain it, and only by those."""
    subsets = _wide_subsets(ghz_32.num_qubits)
    samples = apedp_sampling(ghz_32, shots=4_000, seed=5)
    wire = 7
    counts: dict[tuple[bool, ...], int] = {}
    for bits, count in samples.counts.items():
        flipped = tuple(not bit if index == wire else bit
                        for index, bit in enumerate(bits))
        counts[flipped] = counts.get(flipped, 0) + count
    faulty = dataclasses.replace(samples, counts=counts)
    block = adjudicate_marginal(faulty, ghz_32, subsets)
    assert block["verdict"] == "INCONSISTENT"
    failing = [row["subset"] for row in block["subsets"] if row["verdict"] != "CONSISTENT"]
    assert failing
    assert all(wire in subset for subset in failing)
    assert all(row["verdict"] == "CONSISTENT"
               for row in block["subsets"] if wire not in row["subset"])


def test_a_subset_with_too_few_expected_counts_abstains():
    """A chi-square there is the wrong test, so it abstains rather than rejecting."""
    target = random_stabilizer_target(24, seed=50_240)
    subset = tuple(range(10))
    samples = apedp_sampling(target, shots=12, seed=6)
    block = adjudicate_marginal(samples, target, [subset])
    row = block["subsets"][0]
    assert row["min_expected"] < MIN_EXPECTED_COUNT
    assert row["verdict"] == "UNRESOLVED"
    assert row["family_p_value"] is None
    assert "expected counts" in row["reason"]
    assert block["verdict"] == "UNRESOLVED"
    assert block["unresolved_subsets"] == [list(subset)]


def test_the_family_correction_is_arithmetic_not_a_flourish(ghz_32):
    """Repeating a subset multiplies its p-value: the record cannot buy a pass by volume."""
    samples = apedp_sampling(ghz_32, shots=4_000, seed=5)
    subset = (0, 1)
    single = adjudicate_marginal(samples, ghz_32, [subset])["subsets"][0]
    repeated = adjudicate_marginal(samples, ghz_32, [subset] * 4)["subsets"][0]
    assert repeated["p_value"] == pytest.approx(single["p_value"])
    assert repeated["family_p_value"] == pytest.approx(min(1.0, 4 * single["p_value"]))
    assert repeated["family_p_value"] >= single["family_p_value"]
    assert FAMILY_ALPHA == 1e-3


def test_a_sample_set_from_another_specification_is_refused(ghz_32):
    other = ghz_target(31)
    samples = apedp_sampling(other, shots=10, seed=1)
    with pytest.raises(ValueError):
        adjudicate_marginal(samples, ghz_32, [(0, 1)])
    with pytest.raises(ValueError):
        adjudicate_marginal(samples, ghz_32, [])
    with pytest.raises(ValueError):
        adjudicate_marginal(samples, ghz_32, [(0, 1)], alpha=0.0)


# ------------------------------------------------------------------ two engines
def test_two_correct_engines_agree_and_a_wrong_one_does_not(ghz_32):
    subsets = _wide_subsets(ghz_32.num_qubits)
    own = apedp_sampling(ghz_32, shots=4_000, seed=5)
    try:
        foreign = stim_sampling(ghz_32, shots=4_000, seed=6)
    except ToolUnavailable as error:
        pytest.skip(f"stim is not installed: {error}")
    pair = marginal_pair(own, foreign, ghz_32, subsets)
    assert pair["verdict"] == "AGREE"
    assert pair["shots_each"] == 4_000
    assert pair["family_size"] == len(subsets)
    for row in pair["subsets"]:
        assert row["verdict"] == "AGREE"
        assert row["family_p_value"] == min(1.0, row["p_value"] * len(subsets))

    # The same engine twice from different seeds is two correct samplers, and a wrong
    # sample set is not silently accepted because it is a sample set.
    counts = {}
    for bits, count in own.counts.items():
        flipped = (not bits[0], *tuple(bits[1:]))
        counts[flipped] = counts.get(flipped, 0) + count
    faulty = dataclasses.replace(own, counts=counts)
    disagree = marginal_pair(faulty, foreign, ghz_32, subsets)
    assert disagree["verdict"] == "DISAGREE"
    assert [row["subset"] for row in disagree["subsets"]
            if row["verdict"] == "DISAGREE"]

    with pytest.raises(ValueError):
        marginal_pair(own, dataclasses.replace(foreign, shots=4_001,
                                               counts=own.counts), ghz_32, subsets)


def test_a_different_graph_state_passes_this_referee():
    """The blind spot, measured: every Z-basis marginal of a graph state is uniform.

    Two graph states on the same wires have the same Z-basis marginals, so samples from
    one are consistent with the other. This is why the marginal referee is reported as a
    statement about the tested projections, and why the record still carries the
    enumerable full-distribution comparison beside it.
    """
    target = ring_graph_target(16)
    other = seeded_graph_target(16, extra_edge_probability=0.6, seed=99)
    samples = apedp_sampling(other, shots=4_000, seed=8)
    block = adjudicate_marginal(dataclasses.replace(samples,
                                                    target_sha256=target.sha256()),
                               target, _wide_subsets(16))
    assert block["verdict"] == "CONSISTENT"
    assert len(block["uniform_subsets"]) == block["family_size"]


def test_a_random_state_has_subsets_with_content_and_subsets_without():
    """A marginal can be uniform or not, and the row says which: the choice matters.

    On this 24-qubit draw the contiguous 10-wire projection is uniform while a scattered
    one supports half its cells, so a design that only tested contiguous blocks could
    have reported a blind spot the state did not have. Both facts are pinned here.
    """
    target = random_stabilizer_target(24, seed=50_240)
    contiguous = exact_marginal(target, tuple(range(10)))
    scattered = exact_marginal(target, (3, 7, 10, 13, 15, 16, 18, 19, 21, 22))
    assert contiguous.cells == 1024 and contiguous.is_uniform
    assert not scattered.is_uniform
    assert scattered.support < scattered.cells
    assert scattered.probability == pytest.approx(1.0 / scattered.support)


def test_the_referee_names_what_it_does_not_decide():
    """A row says which subsets and which width, not 'the state is right'."""
    target = ghz_target(20)
    samples = apedp_sampling(target, shots=500, seed=1)
    block = adjudicate_marginal(samples, target, [(0, 1), (2, 3)])
    assert block["referee"] == "exact Z-basis marginals of the specification"
    assert "Bonferroni" in block["test"]
    assert block["alpha"] == FAMILY_ALPHA
    assert all(row["subset_size"] <= 10 for row in block["subsets"])
    assert block["counts_sha256"] == samples.counts_sha256()
