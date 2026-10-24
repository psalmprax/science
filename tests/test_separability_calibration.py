"""Tests for separability analyzer and coupling residual calibration suite."""
from __future__ import annotations

import numpy as np

from apedp.separability_calibration import (
    compute_coupling_residual_ratio,
    generate_calibration_dataset,
    run_full_calibration_experiment,
    run_single_calibration_trial,
)


def test_generate_calibration_dataset_regimes():
    rng = np.random.default_rng(101)
    for regime in ["exact_separable", "weak_interaction", "strong_interaction", "rational_interaction", "3var_separable"]:
        _data, y, is_interactive = generate_calibration_dataset(regime, n=30, noise_level=0.01, rng=rng)
        assert len(y) == 30
        assert np.all(np.isfinite(y))
        if "separable" in regime:
            assert is_interactive is False
        else:
            assert is_interactive is True


def test_coupling_residual_ratio_discriminates_interactions():
    rng = np.random.default_rng(202)
    # Purely separable
    data_sep, y_sep, _ = generate_calibration_dataset("exact_separable", n=50, noise_level=0.0, rng=rng)
    ratio_sep = compute_coupling_residual_ratio(data_sep, y_sep)
    assert ratio_sep < 0.05, f"Separable model ratio should be near zero, got {ratio_sep}"

    # Strong interaction
    data_int, y_int, _ = generate_calibration_dataset("strong_interaction", n=50, noise_level=0.0, rng=rng)
    ratio_int = compute_coupling_residual_ratio(data_int, y_int)
    assert ratio_int > ratio_sep, f"Interactive ratio {ratio_int} should exceed separable ratio {ratio_sep}"
    assert ratio_int > 0.06, f"Interactive model ratio should exceed 0.06, got {ratio_int}"


def test_single_calibration_trial():
    res = run_single_calibration_trial("weak_interaction", noise_level=0.02, scale_ratio=1.0, seed=42)
    assert res.is_interactive is True
    assert np.isfinite(res.unexplained_std_ratio)


def test_calibration_experiment_execution():
    res = run_full_calibration_experiment(
        thresholds=[0.05, 0.10, 0.15, 0.20],
        n_trials_per_cell=2,
        base_seed=123,
    )
    assert res["total_trials"] > 0
    assert len(res["threshold_metrics"]) == 4
    assert res["recommended_threshold"] in [0.05, 0.10, 0.15, 0.20]
    prov = res["provisional_threshold_0_15_metrics"]
    assert prov is not None
    assert 0.0 <= prov["accuracy"] <= 1.0
