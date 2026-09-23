"""Tests for neural discovery validation and benchmarks."""

import jax
import jax.numpy as jnp
import numpy as np

from apedp.neural_discovery.architectures import MLPResNet, init_model
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.neural_discovery.uncertainty import ConformalPredictor
from apedp.neural_discovery.validation import (
    CalibrationSuite,
    ComparisonBaseline,
    OODBenchmarkSuite,
    SRBenchEvaluator,
    run_full_benchmark,
)


def test_ood_covariate_shift():
    """Test covariate shift benchmark."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 1), key, config)

    # Train on [0, 1], test on [1, 2], [2, 3]
    def data_fn(x):
        return x ** 2  # Ground truth: y = x^2

    ood = OODBenchmarkSuite(config)
    results = ood.covariate_shift(
        model, params,
        train_range=(0, 1),
        test_ranges=[(1, 2), (2, 3)],
        data_fn=data_fn,
        n_points=20,
    )

    assert len(results) == 2
    for r in results:
        assert 'mse' in r.metrics
        assert 'r2' in r.metrics
        assert 'extrapolation_factor' in r.metrics
        print(f"{r.name}: MSE={r.metrics['mse']:.4f}, R²={r.metrics['r2']:.4f}, "
              f"extrap={r.metrics['extrapolation_factor']:.2f}")


def test_ood_concept_shift():
    """Test concept shift benchmark."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 1), key, config)

    def data_fn(x, regime):
        # Different regimes: linear vs quadratic
        if regime.get('type') == 'linear':
            return regime['slope'] * x
        return regime['coeff'] * x ** 2

    ood = OODBenchmarkSuite(config)
    results = ood.concept_shift(
        model, params,
        regime_params=[
            {'type': 'linear', 'slope': 2.0, 'x_min': 0, 'x_max': 5},
            {'type': 'quadratic', 'coeff': 1.0, 'x_min': 0, 'x_max': 5},
        ],
        data_fn=data_fn,
        n_points=20,
    )

    assert len(results) == 2
    for r in results:
        assert 'mse' in r.metrics
        assert 'r2' in r.metrics
        assert 'regime' in r.metrics
        print(f"{r.name}: MSE={r.metrics['mse']:.4f}, R²={r.metrics['r2']:.4f}")


def test_ood_dataset_shift():
    """Test dataset shift benchmark."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 1), key, config)

    ood = OODBenchmarkSuite(config)

    # Source data
    x_src = np.linspace(0, 1, 20).reshape(-1, 1)
    y_src = x_src ** 2

    # Target datasets
    x_tgt1 = np.linspace(0, 1, 20).reshape(-1, 1)
    y_tgt1 = x_tgt1 ** 2 + 0.1 * np.random.randn(20, 1)

    x_tgt2 = np.linspace(0, 1, 20).reshape(-1, 1)
    y_tgt2 = 2 * x_tgt2  # Different function

    results = ood.dataset_shift(
        model, params,
        source_data=(x_src, y_src),
        target_datasets=[(x_tgt1, y_tgt1), (x_tgt2, y_tgt2)],
        dataset_names=["noisy_same", "different_fn"],
    )

    assert len(results) == 2
    for r in results:
        assert 'mse' in r.metrics
        assert 'r2' in r.metrics
        print(f"{r.name}: MSE={r.metrics['mse']:.4f}, R²={r.metrics['r2']:.4f}")


def test_calibration_suite():
    """Test CalibrationSuite."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    # Test data
    x_test = jnp.ones((50, 5))
    y_test = jnp.ones((50, 1)) * 2.0 + jax.random.normal(key, (50, 1)) * 0.2

    cal_suite = CalibrationSuite(config)

    # Test with ensemble
    ensemble_params = [params for _ in range(3)]
    result = cal_suite.evaluate(model, params, (x_test, y_test), method="ensemble",
                                ensemble_params=ensemble_params)

    assert 'ece' in result.metrics
    assert 'nll' in result.metrics
    assert 'sharpness' in result.metrics
    for level in [50, 90, 95, 99]:
        assert f'coverage_{level}' in result.metrics
    print(f"ECE: {result.metrics['ece']:.4f}, NLL: {result.metrics['nll']:.4f}")
    for level in [50, 90, 95, 99]:
        print(f"  Coverage {level}%: {result.metrics[f'coverage_{level}']:.3f}")


def test_conformal_calibration():
    """Test calibration with conformal predictor."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    # Calibration data
    x_cal = jnp.ones((30, 5))
    y_cal = jnp.ones((30, 1)) * 2.0 + jax.random.normal(key, (30, 1)) * 0.1

    conformal = ConformalPredictor(alpha=0.1)
    conformal.calibrate(model, params, x_cal, y_cal)

    # Test data
    x_test = jnp.ones ((20, 5))
    y_test = jnp.ones((20, 1)) * 2.0 + jax.random.normal(key, (20, 1)) * 0.1

    cal_suite = CalibrationSuite(config)
    result = cal_suite.evaluate(model, params, (x_test, y_test), method="conformal",
                                conformal=conformal)

    assert 'ece' in result.metrics
    print(f"Conformal ECE: {result.metrics['ece']:.4f}")


def test_srbench_evaluator():
    """Test SRBenchEvaluator (Feynman equations)."""
    config = NeuralDiscoveryConfig()
    evaluator = SRBenchEvaluator(config, noise_level=0.0)

    # Test data generation
    x_data, y_data = evaluator.generate_data("I.6.2", n_samples=10)
    assert len(x_data) == 2  # x1, x2
    assert len(y_data) == 10
    print(f"Generated I.6.2: x1={x_data['x1'][:3]}, y={y_data[:3]}")

    # Test with more equations
    for name in ["I.6.2", "I.10.7", "II.10.15"]:
        x_data, y_data = evaluator.generate_data(name, n_samples=5)
        print(f"  {name}: {len(x_data)} vars, {len(y_data)} samples")


def test_comparison_baseline():
    """Test ComparisonBaseline."""
    config = NeuralDiscoveryConfig()
    baseline = ComparisonBaseline(config)

    # Synthetic data
    np.random.seed(42)
    x_train = np.random.randn(50, 5)
    y_train = x_train[:, 0] * 2 + x_train[:, 1] * -1 + 0.1 * np.random.randn(50)
    x_test = np.random.randn(20, 5)
    y_test = x_test[:, 0] * 2 + x_test[:, 1] * -1 + 0.1 * np.random.randn(20)

    # sklearn MLP
    result = baseline.sklearn_mlp(x_train, y_train, x_test, y_test)
    assert 'mse' in result
    assert 'r2' in result
    assert result['method'] == 'sklearn_mlp'
    print(f"sklearn MLP: MSE={result['mse']:.4f}, R²={result['r2']:.4f}")

    # XGBoost (may not be installed)
    result = baseline.xgboost_baseline(x_train, y_train, x_test, y_test)
    print(f"XGBoost: MSE={result.get('mse', 'N/A')}, R²={result.get('r2', 'N/A')}")

    # All baselines
    results = baseline.run_all_baselines(x_train, y_train, x_test, y_test)
    assert len(results) >= 1


def test_run_full_benchmark():
    """Test run_full_benchmark integration."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 3), key, config)

    # Benchmark data
    np.random.seed(42)
    benchmark_data = {
        'dataset1': (np.random.randn(20, 3), np.random.randn(20)),
        'dataset2': (np.random.randn(20, 3), np.random.randn(20)),
    }

    report = run_full_benchmark(config, model, params, benchmark_data)

    assert len(report.results) == 2
    assert 'mean_r2' in report.summary
    assert 'mean_ece' in report.summary
    assert report.summary['n_datasets'] == 2
    print(f"Full benchmark: mean_r2={report.summary['mean_r2']:.4f}, "
          f"mean_ece={report.summary['mean_ece']:.4f}")


if __name__ == "__main__":
    test_ood_covariate_shift()
    test_ood_concept_shift()
    test_ood_dataset_shift()
    test_calibration_suite()
    test_conformal_calibration()
    test_srbench_evaluator()
    test_comparison_baseline()
    test_run_full_benchmark()
    print("All validation tests passed!")