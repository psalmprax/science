"""Tests for neural discovery integration."""

import jax
import numpy as np

from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.neural_discovery.integration import (
    ModelDiscoveryIntegration,
    NeuralDiscoveryEngine,
    NeuralDiscoveryResult,
    distill_via_surrogate,
    neural_relevance,
)


def test_neural_discovery_engine():
    """Test NeuralDiscoveryEngine basic fit."""
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(16,),
        num_layers=1,
        max_epochs=5,
    )
    engine = NeuralDiscoveryEngine(config)

    # Synthetic data
    np.random.seed(42)
    x = np.random.randn(50, 5).astype(np.float32)
    y = (x[:, 0] * 2 + x[:, 1] * -1 + 0.1 * np.random.randn(50)).astype(np.float32)

    key = jax.random.PRNGKey(42)
    result = engine.fit(x, y, key=key, verbose=False)

    assert isinstance(result, NeuralDiscoveryResult)
    assert result.model is not None
    assert result.params is not None
    assert result.trainer is not None
    assert result.history is not None
    print(f"Engine fit: train_loss={result.history.train_loss[-1]:.4f}, "
          f"val_loss={result.history.val_loss[-1]:.4f}")


def test_neural_discovery_predict():
    """Test NeuralDiscoveryEngine predict_with_uncertainty."""
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(16,),
        num_layers=1,
        max_epochs=5,
        dropout_rate=0.1,
        dropout_mc_samples=10,
    )
    engine = NeuralDiscoveryEngine(config)

    np.random.seed(42)
    x = np.random.randn(50, 5).astype(np.float32)
    y = (x[:, 0] * 2 + 0.1 * np.random.randn(50)).astype(np.float32)

    key = jax.random.PRNGKey(42)
    result = engine.fit(x, y, key=key, verbose=False)

    # Predict with uncertainty
    x_test = np.random.randn(10, 5).astype(np.float32)
    preds = engine.predict_with_uncertainty(result, x_test)

    assert 'mean' in preds
    assert 'std' in preds
    assert 'lower' in preds
    assert 'upper' in preds
    assert preds['mean'].shape == (10, 1)
    assert preds['std'].shape == (10, 1)
    print(f"Predictions: mean={preds['mean'][:3].ravel()}, std={preds['std'][:3].ravel()}")


def test_neural_discovery_symbolic():
    """Test NeuralDiscoveryEngine discover_symbolic."""
    config = NeuralDiscoveryConfig(
        architecture="symbolic_transformer",
        transformer_dim=16,
        transformer_heads=2,
        transformer_layers=1,
        max_seq_len=16,
        vocab_size=30,
        max_epochs=2,
        symbolic_rl_epochs=2,
    )
    engine = NeuralDiscoveryEngine(config)

    x_data = {
        'x0': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3),
        'x1': np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0] * 3),
    }
    y_data = np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0] * 3)

    result = engine.discover_symbolic(x_data, y_data, n_iterations=2)

    assert 'params' in result
    assert 'expressions' in result
    print(f"Discovered: {result['expressions']}")


def test_neural_relevance():
    """Test neural_relevance function."""
    # Constructed to exercise config validation; the relevance call below
    # takes no config.
    _config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(16,),
        num_layers=1,
        max_epochs=3,
    )

    np.random.seed(42)
    x_data = {
        'a': np.random.randn(40).astype(np.float32),
        'b': np.random.randn(40).astype(np.float32),
        'c': np.random.randn(40).astype(np.float32),
    }
    y = (x_data['a'] * 2 + x_data['c'] * -1 + 0.1 * np.random.randn(40)).astype(np.float32)

    relevance = neural_relevance(x_data, y, seed=42)

    assert 'a' in relevance or 'relevant' in relevance
    print(f"Relevance: {relevance}")


def test_distill_via_surrogate():
    """Test distill_via_surrogate function."""
    x_data = {
        'x0': np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4),
        'x1': np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4),
    }
    y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0] * 4)  # 2 * x0

    result = distill_via_surrogate(x_data, y_data, seed=42, n_dense=50)

    assert 'terms' in result or 'expression' in result
    print(f"Result: {result}")


def test_model_discovery_integration():
    """Test ModelDiscoveryIntegration (requires ModelDiscoveryEngine)."""
    # This is a smoke test - create the integration object
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(16,),
        num_layers=1,
        max_epochs=2,
    )
    # Constructing the engine is the smoke check; it is not driven further.
    _neural_engine = NeuralDiscoveryEngine(config)

    # Mock model discovery engine (would import from model_discovery in real use)
    class MockModelDiscoveryEngine:
        pass

    _mock_engine = MockModelDiscoveryEngine()
    integration = ModelDiscoveryIntegration(config=config)

    assert integration.engine is not None
    assert integration.config is config

    # Test neural_relevance through integration
    np.random.seed(42)
    x_data = {
        'x0': np.random.randn(20).astype(np.float32),
        'x1': np.random.randn(20).astype(np.float32),
        'x2': np.random.randn(20).astype(np.float32),
    }
    y = (x_data['x0'] * 2 + 0.1 * np.random.randn(20)).astype(np.float32)

    relevance = integration.engine.neural_relevance(x_data, y)
    assert 'relevant' in relevance or 'relevance' in relevance
    print(f"Integration relevance: {relevance}")


if __name__ == "__main__":
    test_neural_discovery_engine()
    test_neural_discovery_predict()
    test_neural_discovery_symbolic()
    test_neural_relevance()
    test_distill_via_surrogate()
    test_model_discovery_integration()
    print("All integration tests passed!")