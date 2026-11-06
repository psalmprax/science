"""Pins for the noise/domain robustness measurement protocol."""

import numpy as np
import pytest

from scripts.measure_search_engine_gaps import (
    add_train_scaled_noise,
    parse_input_range,
    sample_law,
)


def test_planted_law_sampling_is_seeded_and_respects_input_range():
    first = sample_law("x0**2", seed=17, n=100, n_vars=1, input_range=(0.1, 10.0))
    second = sample_law("x0**2", seed=17, n=100, n_vars=1, input_range=(0.1, 10.0))

    np.testing.assert_array_equal(first[0][0], second[0][0])
    np.testing.assert_array_equal(first[1], second[1])
    assert 0.1 <= first[0][0].min()
    assert first[0][0].max() < 10.0
    assert parse_input_range("0.1:10") == (0.1, 10.0)
    with pytest.raises(ValueError, match="increasing pair"):
        sample_law("x0", seed=17, n=10, n_vars=1, input_range=(2.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        sample_law("x0", seed=17, n=10, n_vars=1, input_range=(-1.0, 2.0))


def test_holdout_targets_cannot_change_training_noise_scale_or_draws():
    train = np.linspace(-1.0, 1.0, 20)
    holdout = np.linspace(-2.0, 2.0, 10)

    train_a, holdout_a, scale_a = add_train_scaled_noise(
        train, holdout, noise_fraction=0.05, seed=3
    )
    train_b, holdout_b, scale_b = add_train_scaled_noise(
        train, holdout + 1_000_000.0, noise_fraction=0.05, seed=3
    )

    assert scale_a == scale_b
    np.testing.assert_array_equal(train_a, train_b)
    assert not np.array_equal(holdout_a, holdout_b)
    with pytest.raises(ValueError, match="finite and non-negative"):
        add_train_scaled_noise(train, holdout, noise_fraction=-0.1, seed=3)
