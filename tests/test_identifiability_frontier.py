"""Tests for Identifiability Frontier and Expanded Adversaries in APEDP-RH."""
import numpy as np

from apedp.mathematical.rh_adversarial import RiemannAdversarialEngine
from apedp.mathematical.rh_dataset import RiemannZetaBenchmarkDataset
from apedp.mathematical.rh_precision import IdentifiabilityFrontierEngine


def test_expanded_fake_rh_adversaries():
    """Verify all 5 new adversarial classes generate valid perturbed datasets."""
    n_arr = np.arange(1, 101, dtype=float)
    gamma_arr = 14.1347 + n_arr * 2.0
    adv = RiemannAdversarialEngine(n_arr, gamma_arr)

    # 1. Local agreement
    w1 = adv.get_adversary_local_agreement(epsilon=1e-4, n_obs_max=50)
    assert len(w1["beta"]) == 100
    assert w1["beta"][0] < 0.5001
    assert w1["beta"][-1] > 0.501  # Diverges outside observed range

    # 2. Delayed divergence
    w2 = adv.get_adversary_delayed_divergence(split_n=50, shift=0.05)
    assert np.all(w2["beta"][:49] == 0.5)
    assert np.all(w2["beta"][50:] == 0.55)

    # 3. Oscillatory cancellation
    w3 = adv.get_adversary_oscillatory_cancellation(epsilon=1e-3)
    assert len(w3["beta"]) == 100
    assert not np.all(w3["beta"] == 0.5)

    # 4. Slowly vanishing deviation
    w4 = adv.get_adversary_slowly_vanishing_deviation()
    assert w4["beta"][0] > w4["beta"][-1]  # Monotonically decaying to 0.5

    # 5. Tiny deviation
    w5 = adv.get_adversary_tiny_deviation(k=8)
    assert np.max(np.abs(w5["beta"] - 0.5)) <= 1e-8


def test_identifiability_frontier_sweep():
    """Verify that identifiability frontier demonstrates UNIQUE -> MULTI_HYPOTHESIS -> UNIDENTIFIABLE."""
    dataset = RiemannZetaBenchmarkDataset(n_total_zeros=100, train_n=70, val_n=30)
    frontier_engine = IdentifiabilityFrontierEngine(
        epsilons=(1e-2, 1e-4, 1e-8, 1e-12, 1e-16, 1e-18),
        floating_point_eps=1e-15,
    )

    report = frontier_engine.map_frontier(base_n=dataset.n_arr, noise_sigma=0.0)

    assert len(report.transitions) == 6
    assert 1e-2 in report.unique_support_regime
    assert 1e-18 in report.unidentifiable_regime
    assert report.critical_resolution_boundary <= 1e-6
    assert "Identifiability Frontier" in report.summary
