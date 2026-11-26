"""Counting independent trials (development, not wired into the referee): the pieces behave as described."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_samples

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD
from apedp_dyn.epistemic import independent_trials as it


def _families(sizes: list[int], rho: float, bars: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = []
    for size in sizes:
        f = rng.standard_normal(bars)
        cols += [np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal(bars) for _ in range(size)]
    return 0.01 * np.column_stack(cols)


def test_the_silhouette_is_scikit_learns() -> None:
    r = _families([5, 5, 5], 0.6)
    dist = it.distance(it._corr(r))
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert np.allclose(it._silhouette(dist, labels), silhouette_samples(dist, labels, metric="precomputed"))


def test_families_of_near_copies_are_found_and_independent_trials_stay_apart() -> None:
    assert it.cluster_labels(_families([10] * 6, 0.8)).max() + 1 == 6
    assert it.cluster_labels(_families([10] * 6, 0.8), method="kmeans", max_k=12, n_init=2).max() + 1 == 6
    assert it.cluster_labels(_families([1] * 40, 0.0, seed=3)).max() + 1 == 40


def test_the_eigenvalue_count_runs_from_one_to_all() -> None:
    assert abs(it.eigen_effective(_families([1] * 30, 0.0)) - 30) < 1e-6
    assert it.eigen_effective(_families([30], 0.999)) < 3


def test_deflated_reports_three_ways_against_the_canonical_threshold() -> None:
    r = _families([10] * 5, 0.8, seed=5)
    d = it.deflated(r)
    assert d["threshold"] == DSR_THRESHOLD and d["trials"] == 50
    assert d["as_referee"]["n_trials"] == 50 and d["by_cluster"]["n_trials"] == 5
    for way in ("as_referee", "by_cluster", "by_eigenvalues"):
        assert 0.0 <= d[way]["dsr"] <= 1.0 and d[way]["significant"] == (d[way]["dsr"] >= DSR_THRESHOLD)


def test_cluster_returns_weights_by_inverse_variance() -> None:
    a = np.array([0.01, -0.01, 0.01, -0.01])
    r = np.column_stack([a, 2 * a, np.full(4, 0.003)])        # variances 1 : 4 : 0
    out = it.cluster_returns(r, np.array([0, 0, 1]))
    assert out.shape == (4, 2)
    assert np.allclose(out[:, 0], 0.8 * a + 0.2 * 2 * a)       # weights 4/5 and 1/5
    assert np.allclose(out[:, 1], 0.003)                       # a constant member keeps its series
