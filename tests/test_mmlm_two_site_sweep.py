"""Numerical checks of two-site empirical-amplitude tensor fitting."""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pytest

from apedp_mmlm import TensorTrainMemoryEngine


def _dense_tensor(engine):
    tensor = engine.dmrg_tensors[0][0]
    for core in engine.dmrg_tensors[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    return tensor[..., 0]


def _target(sequences, symbols):
    target = np.zeros((len(symbols),) * len(sequences[0]))
    indices = {token: index for index, token in enumerate(symbols)}
    for seq, count in Counter(tuple(seq) for seq in sequences).items():
        target[tuple(indices[token] for token in seq)] = math.sqrt(count / len(sequences))
    return target


def test_sweep_recovers_known_amplitudes_and_schmidt_entropy():
    sequences = [[2, 2, 2, 2]] * 3 + [[7, 7, 7, 7]]
    engine = TensorTrainMemoryEngine(vocab_size=16, bond_dim=6)
    original_cores = engine.cores.copy()
    original_probabilities = engine.predict_next_proba([2, 7])
    result = engine.dmrg_two_site_sweep(sequences, n_sweeps=2, max_bond_dim=2)
    target = _target(sequences, engine.dmrg_token_ids)
    fitted = _dense_tensor(engine)
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    assert engine.dmrg_token_ids == (2, 7)
    assert all(core.shape[1] == 2 for core in engine.dmrg_tensors)
    assert result["sweeps_completed"] == 2
    expected_entropy = -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    assert result["mean_entanglement_entropy"] == pytest.approx(expected_entropy)
    assert result["max_entanglement_entropy"] == pytest.approx(expected_entropy)
    np.testing.assert_allclose(result["top_schmidt_spectrum"], np.sqrt([0.75, 0.25]))
    assert result["squared_error_by_sweep"][-1] == pytest.approx(
        float(np.sum((fitted - target) ** 2)), abs=1e-12
    )
    np.testing.assert_array_equal(engine.cores, original_cores)
    np.testing.assert_array_equal(engine.predict_next_proba([2, 7]), original_probabilities)


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_rank_limits_and_errors_match_dense_reconstruction(rank):
    sequences = [list(seq) for seq in itertools.product((0, 1), repeat=4) if sum(seq) % 2 == 0]
    engine = TensorTrainMemoryEngine(4, bond_dim=4)
    report = engine.dmrg_two_site_sweep(sequences, n_sweeps=3, max_bond_dim=rank)
    target = _target(sequences, engine.dmrg_token_ids)
    assert all(1 <= dim <= rank for dim in report["bond_dimensions"])
    assert report["squared_error_by_sweep"][-1] == pytest.approx(
        float(np.sum((_dense_tensor(engine) - target) ** 2)), abs=1e-12
    )
    assert all(after <= before + 1e-12 for before, after in itertools.pairwise(
        report["squared_error_by_sweep"]
    ))
    assert -1e-12 <= report["max_entanglement_entropy"] <= math.log(rank) + 1e-12
    assert sum(value ** 2 for value in report["top_schmidt_spectrum"]) == pytest.approx(1)
    if rank >= 2:
        np.testing.assert_allclose(_dense_tensor(engine), target, atol=1e-12)


def test_relative_cutoff_discards_small_schmidt_component():
    sequences = [[0, 0]] * 99 + [[1, 1]]
    engine = TensorTrainMemoryEngine(4)
    report = engine.dmrg_two_site_sweep(sequences, max_bond_dim=4, cutoff=0.2)
    assert report["bond_dimensions"] == [1]
    assert report["squared_error_by_sweep"][-1] == pytest.approx(0.01)
    assert report["max_entanglement_entropy"] == pytest.approx(0)


def test_fitting_is_deterministic_and_single_symbol_is_supported():
    engine = TensorTrainMemoryEngine(4)
    report = engine.dmrg_two_site_sweep([[3, 3, 3]] * 2)
    assert report["bond_dimensions"] == [1, 1]
    np.testing.assert_allclose(_dense_tensor(engine), np.ones((1, 1, 1)))
    first_tensors = [core.copy() for core in engine.dmrg_tensors]
    assert report == engine.dmrg_two_site_sweep([[3, 3, 3]] * 2)
    for first, second in zip(first_tensors, engine.dmrg_tensors):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("sequences, options, message", [
    ([], {}, "at least one"),
    ([[1]], {}, "length >= 2"),
    ([[1, 2], [1, 2, 3]], {}, "equal-length"),
    ([[1, -1]], {}, "integer IDs"),
    ([[1, 4]], {}, "integer IDs"),
    ([[1, 1.5]], {}, "integer IDs"),
    ([[1, 2]], {"n_sweeps": 0}, "positive"),
    ([[1, 2]], {"n_sweeps": 1.5}, "positive integers"),
    ([[1, 2]], {"n_sweeps": True}, "positive integers"),
    ([[1, 2]], {"max_bond_dim": 0}, "positive"),
    ([[1, 2]], {"max_bond_dim": 1.5}, "positive integers"),
    ([[1, 2]], {"cutoff": float("nan")}, "cutoff"),
    ([[1, 2]], {"cutoff": -1}, "cutoff"),
    ([[1, 2]], {"cutoff": 1}, "cutoff"),
])
def test_invalid_fits_fail_explicitly_without_changing_the_engine(sequences, options, message):
    engine = TensorTrainMemoryEngine(4)
    cores = engine.cores.copy()
    with pytest.raises(ValueError, match=message):
        engine.dmrg_two_site_sweep(sequences, **options)
    assert engine.dmrg_tensors == []
    assert engine.dmrg_token_ids == ()
    np.testing.assert_array_equal(engine.cores, cores)
