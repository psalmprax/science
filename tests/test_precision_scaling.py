"""Tests for Precision-Scaling Identifiability Experiment."""
from apedp.mathematical.precision_scaling_experiment import PrecisionScalingEngine


def test_precision_scaling_engine_transitions():
    """Verify that empirical resolution threshold scales with arithmetic precision."""
    engine = PrecisionScalingEngine(n_points=50, seed=42)

    # float32 sweep
    r32 = engine.run_float32_sweep(epsilons=[1e-2, 1e-3, 1e-4, 1e-6])
    assert r32.empirical_epsilon_star >= 1e-4
    assert "float32" in r32.regime_name

    # float64 sweep
    r64 = engine.run_float64_sweep(epsilons=[1e-4, 1e-6, 1e-8, 1e-12])
    assert r64.empirical_epsilon_star <= 1e-6
    assert r64.empirical_epsilon_star >= 1e-10

    # mpmath 50-digit sweep
    rmp = engine.run_mpmath_sweep(dps=50, epsilons=[1e-5, 1e-15, 1e-25, 1e-35])
    assert rmp.empirical_epsilon_star <= 1e-15

    # Confirm the threshold moved across regimes
    assert r32.empirical_epsilon_star > r64.empirical_epsilon_star > rmp.empirical_epsilon_star
