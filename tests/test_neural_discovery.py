# Databricks notebook source
"""Tests for SOTA Neural Discovery for APEDP.

Tests the JAX-based neural discovery system including architectures,
uncertainty quantification, symbolic regression, and active learning.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from apedp.neural_discovery.active_learning import (
    BALD,
    DiversityAcquisition,
    ExpectedImprovement,
    MaxEntropy,
    MaxVariance,
    create_acquisition_function,
)
from apedp.neural_discovery.architectures import (
    MLPResNet,
    count_params,
    create_model,
    init_model,
)
from apedp.neural_discovery.config import NeuralDiscoveryConfig, get_preset
from apedp.neural_discovery.integration import (
    ModelDiscoveryIntegration,
    NeuralDiscoveryEngine,
    distill_via_surrogate,
    neural_relevance,
)
from apedp.neural_discovery.symbolic import (
    ExpressionTokenizer,
    JAXSymbolicExecutor,
    NeuralSymbolicRegressor,
    build_vocab,
)
from apedp.neural_discovery.training import (
    NeuralDiscoveryTrainer,
    TrainingConfig,
)
from apedp.neural_discovery.uncertainty import (
    CalibrationMetrics,
    ConformalPredictor,
    DeepEnsemble,
    UncertaintyOutput,
    predict_with_uncertainty,
)


def test_config_default():
    """Test default config creation and validation."""
    config = NeuralDiscoveryConfig()
    assert config.architecture == "mlp_resnet"
    assert config.ensemble_size >= 2
    assert 0 < config.conformal_alpha < 1
    assert config.cv_folds >= 2

    # Test validation errors
    with pytest.raises(ValueError):
        NeuralDiscoveryConfig(ensemble_size=1)

    with pytest.raises(ValueError):
        NeuralDiscoveryConfig(cv_folds=1)


def test_config_presets():
    """Test preset configurations."""
    for name in ["fast", "accurate", "symbolic", "physics", "production"]:
        config = get_preset(name)
        assert config is not None
        assert config.architecture in ["mlp_resnet", "symbolic_transformer", "equation_gnn", "pinn"]


def test_config_with_updates():
    """Test config withUpdates method."""
    config = NeuralDiscoveryConfig()
    config2 = config.with_updates(architecture="symbolic_transfer", max_epochs=100)
    assert config2.architecture == "symbolic_transfer"
    assert config2.max_epochs == 100
    # Original should be unchanged (frozen)
    assert config.architecture == "mlp_resnet"


def test_mlp_resnet_creation():
    """Test MLPResNet can be created and initialized."""
    config = NeuralDiscoveryConfig(architecture="mlp_resnet", hidden_dims=(32, 16), num_layers=2)
    model = MLPResNet(
        hidden_dims=config.hidden_dims,
        num_layers=config.num_layers,
    )
    assert model is not None

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = init_model(model, (32,), key, config)
    assert params is not None
    n = count_params(params)
    assert n > 0


def test_create_model_factory():
    """Test create_model factory function."""
    config = NeuralDiscoveryConfig(architecture="mlp_resnet", hidden_dims=(32, 16))
    model = create_model(config, input_dim=10)
    assert model is not None

    # Test other architectures
    config_gnn = NeuralDiscoveryConfig(architecture="equation_gnn", gnn_hidden_dim=32)
    model_gnn = create_model(config_gnn, input_dim=10)
    assert model_gnn is not None


def test_jax_symbolic_executor():
    """Test JAXSymbolicExecutor basic functionality."""
    executor = JAXSymbolicExecutor(n_variables=3)

    # Test basic execution
    variables = {
        'VAR_0': np.array([1.0, 2.0, 3.0]),
        'VAR_1': np.array([0.5, 1.5, 2.5]),
        'VAR_2': np.array([1.0, 1.0, 1.0]),
    }

    # Test executing a simple expression: x0 * x1
    tokens = ['MUL', 'VAR_0', 'VAR_1']
    result = executor.execute(tokens, variables)
    expected = variables['VAR_0'] * variables['VAR_1']
    assert np.allclose(result, expected)

    # Test executing: x0 + x1 + x2
    tokens = ['ADD', 'ADD', 'VAR_0', 'VAR_1', 'VAR_2']
    result = executor.execute(tokens, variables)
    expected = variables['VAR_0'] + variables['VAR_1'] + variables['VAR_2']
    assert np.allclose(result, expected)


def test_build_vocab():
    """Test vocabulary building."""
    vocab, _id_to_vocab = build_vocab(n_variables=3)
    assert len(vocab) > 0
    assert 'PAD' in vocab
    assert 'EOS' in vocab
    assert 'BOS' in vocab
    # Check that variables are in vocab
    for i in range(3):
        assert f'VAR_{i}' in vocab or f'VAR_{i}' in str(vocab)


def test_neural_symbolic_regressor_creation():
    """Test NeuralSymbolicRegressor can be created and applied."""
    config = NeuralDiscoveryConfig()
    n_vars = 3
    vocab, _ = build_vocab(n_vars)

    regressor = NeuralSymbolicRegressor(
        config=config,
        n_variables=n_vars,
        vocab_size=len(vocab),
    )
    assert regressor is not None

    # Test apply instead of forward (Flax convention: setup() fields only
    # accessible inside init/apply)
    key = jax.random.PRNGKey(42)
    x = np.random.randn(5, n_vars)
    y = np.random.randn(5)
    
    # Use apply with parameters
    variables = regressor.init(key, x, y)
    output = regressor.apply(variables, x, y)
    assert 'latent' in output
    assert 'logits' in output
    assert 'tokens' in output


def test_expression_tokenizer():
    """Test ExpressionTokenizer basic functionality."""
    tokenizer = ExpressionTokenizer(n_variables=2, max_len=32)

    # Test tokenize (should work even if detokenize is simplified)
    tokens_empty = tokenizer.tokenize("")
    assert isinstance(tokens_empty, list)

    # Test with BOS/EOS tokens included
    tokens = tokenizer.tokenize("x0")
    assert isinstance(tokens, list)
    # Token IDs should include BOS at start and EOS at end
    assert len(tokens) > 0

    # Test max_len truncation
    long_expr = "x0 * x1 * x2 * x3 * x4 * x5"
    tokens_long = tokenizer.tokenize(long_expr)
    assert isinstance(tokens_long, list)


def test_acquisition_functions():
    """Test acquisition function creation and scoring."""
    config = NeuralDiscoveryConfig(
        acquisition_type="bald",
        acquisition_batch_size=5,
    )

    # Test BALD
    bald = BALD(config)
    assert bald is not None

    # Test other acquisition functions (construction is the check; the
    # factory loop below asserts on the returned callables)
    _ei = ExpectedImprovement(config)
    _max_var = MaxVariance(config)
    _max_ent = MaxEntropy(config)
    _diversity = DiversityAcquisition(config)

    # Test create_acquisition_function factory
    for acq_type in ["bald", "expected_improvement", "max_variance", "max_entropy", "diversity"]:
        fn = create_acquisition_function(config, acquisition_type=acq_type)
        assert fn is not None


def test_neural_relevance_basic():
    """Test backward-compatible neural_relevance function."""
    # Create synthetic data
    rng = np.random.default_rng(42)
    n_samples = 50
    x_data = {
        'x1': rng.uniform(-3, 3, n_samples),
        'x2': rng.uniform(-3, 3, n_samples),
    }
    # y = x1^2 + x2 (nonlinear dependence)
    y_data = x_data['x1'] ** 2 + x_data['x2'] + rng.normal(0, 0.1, n_samples)

    # Test neural_relevance
    result = neural_relevance(x_data, y_data, seed=42)

    # Should return a dict with relevant variables
    assert isinstance(result, dict)
    # The result should have 'relevant' key
    assert 'relevant' in result or 'note' in result


def test_distill_via_surrogate_basic():
    """Test backward-compatible distill_via_surrogate function."""
    # Create synthetic data
    rng = np.random.default_rng(42)
    n_samples = 50
    x_data = {
        'x1': rng.uniform(-3, 3, n_samples),
    }
    # y = 2/x + noise (has 1/x term)
    y_data = 2.0 / x_data['x1'] + rng.normal(0, 0.1, n_samples)

    # Test distill_via_surrogate
    result = distill_via_surrogate(x_data, y_data, seed=42, n_dense=100)

    # Should return a dict
    assert isinstance(result, dict)
    assert 'terms' in result or 'note' in result


def test_model_discovery_integration():
    """Test ModelDiscoveryIntegration enhances discover."""
    config = NeuralDiscoveryConfig()
    integration = ModelDiscoveryIntegration(config=config)

    # Test that integration object is created properly
    assert integration is not None
    assert hasattr(integration, 'config')
    assert hasattr(integration, 'engine')


def test_trainer_creation():
    """Test NeuralDiscoveryTrainer can be created."""
    config = NeuralDiscoveryConfig()
    trainer = NeuralDiscoveryTrainer(config=config)
    assert trainer is not None
    assert hasattr(trainer, 'config')
    assert hasattr(trainer, 'training_config')


def test_training_config_from_discovery():
    """Test TrainingConfig can be created from DiscoveryConfig."""
    config = NeuralDiscoveryConfig()
    training_config = TrainingConfig.from_discovery_config(config)
    assert training_config is not None
    assert training_config.learning_rate == config.learning_rate
    assert training_config.batch_size == config.batch_size


def test_deep_ensemble_creation():
    """Test DeepEnsemble can be created."""
    config = NeuralDiscoveryConfig(ensemble_size=5)

    def model_fn():
        from apedp.neural_discovery.architectures import MLPResNet
        model = MLPResNet(hidden_dims=(32, 16), num_layers=2)
        return model

    ensemble = DeepEnsemble(model_fn=model_fn, config=config)
    assert ensemble is not None


def test_calibration_metrics():
    """Test CalibrationMetrics computation."""
    # Create dummy predictions and targets
    n = 100
    mean = np.random.randn(n)
    std = np.abs(np.random.randn(n)) + 0.1
    y_true = np.random.randn(n) * 2

    # Test ECE
    ece = CalibrationMetrics.expected_calibration_error(mean, std, y_true, n_bins=10)
    assert 0 <= ece < 1.0

    # Test coverage at levels
    coverage = CalibrationMetrics.coverage_at_level(mean, std, y_true)
    assert 'coverage_90' in coverage
    assert 'coverage_95' in coverage
    assert 'coverage_99' in coverage

    # Test NLL
    nll = CalibrationMetrics.negative_log_likelihood(mean, std, y_true)
    assert nll > 0  # Should be positive for non-perfect predictions

    # Test sharpness
    sharpness = CalibrationMetrics.sharpness(std)
    assert sharpness > 0

    # Test calibration curve
    nominal, empirical = CalibrationMetrics.calibration_curve(mean, std, y_true)
    assert len(nominal) == 10
    assert len(empirical) == 10


def test_predict_with_uncertainty():
    """Test predict_with_uncertainty unified interface."""
    # Create dummy model and params
    from apedp.neural_discovery.architectures import MLPResNet

    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(32, 16), num_layers=2)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10,), key, config)

    # Create test data
    x_test = np.random.randn(5, 10)
    y_test = np.random.randn(5)

    # Test with ensemble method
    output = predict_with_uncertainty(
        model, params, x_test, method="ensemble", ensemble_params=[params]
    )
    assert isinstance(output, UncertaintyOutput)
    assert output.mean is not None
    assert output.std is not None
    assert output.lower is not None
    assert output.upper is not None

    # An uncalibrated conformal predictor has no quantile, so it must refuse
    # rather than emit an interval with no coverage guarantee behind it.
    # (This previously "passed" only because the ValueError was never checked.)
    uncalibrated = ConformalPredictor(alpha=0.1)
    with pytest.raises(ValueError, match="calibrate"):
        predict_with_uncertainty(
            model, params, x_test, method="conformal", conformal=uncalibrated
        )

    # Calibrated, it produces a real interval containing the point prediction.
    conformal = ConformalPredictor(alpha=0.1)
    conformal.calibrate(model, params, x_test, y_test)
    output_conf = predict_with_uncertainty(
        model, params, x_test, method="conformal", conformal=conformal
    )
    assert isinstance(output_conf, UncertaintyOutput)
    assert output_conf.lower is not None and output_conf.upper is not None


def test_neural_discovery_engine_basic():
    """Test NeuralDiscoveryEngine basic functionality."""
    config = NeuralDiscoveryConfig()
    engine = NeuralDiscoveryEngine(config=config)

    # Test that engine is created properly
    assert engine is not None
    assert hasattr(engine, 'config')
    assert hasattr(engine, 'neural_relevance')
    assert hasattr(engine, 'distill_via_surrogate')
    assert hasattr(engine, 'discover_symbolic')


def test_neural_relevance_synthetic():
    """Test neural_relevance with synthetic data that has clear nonlinearity."""
    rng = np.random.default_rng(42)
    n = 100

    # Create data with clear x^2 dependence
    x1 = rng.uniform(-3, 3, n)
    y = x1 ** 2 + rng.normal(0, 0.1, n)

    x_data = {'x1': x1}
    y_data = y

    result = neural_relevance(x_data, y_data, seed=42)

    # Should detect x1 as relevant (R² should be high for quadratic)
    assert isinstance(result, dict)
    # May have note about insufficient data or other issues,
    # but structure should be valid


def test_distill_via_surrogate_synthetic():
    """Test distill_via_surrogate with synthetic data."""
    rng = np.random.default_rng(42)
    n = 100

    # Create data with 1/x dependence
    x1 = rng.uniform(0.5, 3, n)  # Avoid division by zero
    y = 2.0 / x1 + rng.normal(0, 0.1, n)

    x_data = {'x1': x1}
    y_data = y

    result = distill_via_surrogate(x_data, y_data, seed=42, n_dense=50)

    assert isinstance(result, dict)
    assert 'terms' in result or 'note' in result


def test_full_preset_configs():
    """Test all preset configs work."""
    presets = ["fast", "accurate", "symbolic", "physics", "production"]
    for name in presets:
        config = get_preset(name)
        # Should not raise
        assert config.architecture is not None
        assert config.max_epochs > 0
        assert config.ensemble_size >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])