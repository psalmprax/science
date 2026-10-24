"""Tests for neural discovery uncertainty quantification."""

import jax
import jax.numpy as jnp
import numpy as np

from apedp.neural_discovery.architectures import MLPResNet, init_model
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.neural_discovery.uncertainty import (
    BayesianNN,
    CalibrationMetrics,
    ConformalPredictor,
    DeepEnsemble,
    UncertaintyOutput,
    predict_with_uncertainty,
)


def test_uncertainty_output():
    """Test UncertaintyOutput dataclass."""
    mean = jnp.array([1.0, 2.0, 3.0])
    std = jnp.array([0.1, 0.2, 0.3])
    output = UncertaintyOutput(
        mean=mean,
        std=std,
        lower=mean - 1.96 * std,
        upper=mean + 1.96 * std,
    )
    assert output.mean.shape == (3,)
    assert output.std.shape == (3,)
    assert output.epistemic is None
    assert output.aleatoric is None


def test_deep_ensemble():
    """Test DeepEnsemble creation and prediction."""
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(32, 16),
        num_layers=2,
        ensemble_size=3,
    )

    def model_fn():
        return MLPResNet(
            hidden_dims=config.hidden_dims,
            num_layers=config.num_layers,
            output_dim=1,
        )

    ensemble = DeepEnsemble(model_fn, config)
    assert ensemble.ensemble_size == 3

    # Test pairwise disagreement
    x = jnp.ones((4, 10))
    # Need to add some fake models/params
    ensemble.models = [model_fn() for _ in range(3)]
    ensemble.params_list = []
    key = jax.random.PRNGKey(42)
    for i, model in enumerate(ensemble.models):
        params = init_model(model, x.shape, jax.random.fold_in(key, i), config)
        ensemble.params_list.append(params)

    # Test disagreement computation
    disagreement = ensemble.pairwise_disagreement(x)
    assert isinstance(disagreement, float)
    assert disagreement >= 0
    print(f"Pairwise disagreement: {disagreement}")


def test_conformal_predictor():
    """Test ConformalPredictor calibration and prediction."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(32,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    # Generate synthetic data
    x_cal = jnp.ones((20, 5))
    y_cal = jnp.ones((20, 1)) * 2.0 + jax.random.normal(key, (20, 1)) * 0.1

    conformal = ConformalPredictor(alpha=0.1)
    cal_info = conformal.calibrate(model, params, x_cal, y_cal)

    assert 'quantile' in cal_info
    assert 'coverage' in cal_info
    assert cal_info['target_coverage'] == 0.9
    print(f"Conformal quantile: {cal_info['quantile']}, coverage: {cal_info['coverage']}")

    # Test prediction
    x_test = jnp.ones((5, 5))
    output = conformal.predict(model, params, x_test)
    assert isinstance(output, UncertaintyOutput)
    assert output.mean.shape == (5, 1)


def test_calibration_metrics():
    """Test calibration metrics computation."""
    np.random.seed(42)
    n = 100
    mean = np.random.randn(n)
    std = np.abs(np.random.randn(n)) * 0.5 + 0.1
    y_true = mean + np.random.randn(n) * std

    # ECE
    ece = CalibrationMetrics.expected_calibration_error(mean, std, y_true)
    assert isinstance(ece, float)
    assert ece >= 0
    print(f"ECE: {ece}")

    # Coverage
    coverage = CalibrationMetrics.coverage_at_level(mean, std, y_true)
    for level in [0.5, 0.9, 0.95, 0.99]:
        key = f'coverage_{int(level*100)}'
        assert key in coverage
        assert 0 <= coverage[key] <= 1
        print(f"  {key}: {coverage[key]:.3f} (target: {level})")

    # NLL
    nll = CalibrationMetrics.negative_log_likelihood(mean, std, y_true)
    assert isinstance(nll, float)
    print(f"NLL: {nll}")

    # Sharpness
    sharp = CalibrationMetrics.sharpness(std)
    assert sharp > 0
    print(f"Sharpness: {sharp}")

    # Calibration curve
    nominal, empirical = CalibrationMetrics.calibration_curve(mean, std, y_true)
    assert len(nominal) == len(empirical)
    print(f"Calibration curve: {len(nominal)} points")


def test_predict_with_uncertainty():
    """Test unified predict_with_uncertainty function."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(32,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    x = jnp.ones((5, 5))

    # Single prediction
    output = predict_with_uncertainty(model, params, x)
    assert output.mean.shape == (5, 1)
    assert jnp.all(output.std == 0)

    # Ensemble (mock)
    ensemble_params = [params for _ in range(3)]
    output = predict_with_uncertainty(model, params, x, method="ensemble", ensemble_params=ensemble_params)
    assert output.mean.shape == (5, 1)
    assert output.std.shape == (5, 1)
    assert output.epistemic is not None


def test_bayesian_nn_init():
    """Test BayesianNN variational parameter initialization."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)

    bayesian_nn = BayesianNN(model, config)
    key = jax.random.PRNGKey(42)
    var_params = bayesian_nn.init_variational_params(key, (4, 10))

    assert 'variational' in var_params
    # Check mean and log_std exist for each param
    for name in var_params['variational']:
        assert name.endswith(('_mean', '_log_std'))

    # Test sampling
    sampled = bayesian_nn.sample_params(var_params, key, n_samples=5)
    assert 'params' in sampled
    print(f"BayesianNN variational params: {len(var_params['variational'])} tensors")


if __name__ == "__main__":
    test_uncertainty_output()
    test_deep_ensemble()
    test_conformal_predictor()
    test_calibration_metrics()
    test_predict_with_uncertainty()
    test_bayesian_nn_init()
    print("All uncertainty tests passed!")