"""Tests for neural discovery active learning."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apedp.neural_discovery.active_learning import (
    BALD,
    ActiveLearningLoop,
    DiversityAcquisition,
    ExpectedImprovement,
    MaxEntropy,
    MaxVariance,
    create_acquisition_function,
)
from apedp.neural_discovery.architectures import MLPResNet, init_model
from apedp.neural_discovery.config import (
    AcquisitionType,
    NeuralDiscoveryConfig,
)


def test_acquisition_function_base():
    """Test base AcquisitionFunction diversity logic."""
    config = NeuralDiscoveryConfig(
        acquisition_batch_size=5,
        acquisition_diversity_weight=0.3,
    )

    # Create a concrete acquisition function
    bald = BALD(config)

    # Test _apply_diversity
    candidates = np.random.randn(20, 5)
    scores = np.random.rand(20)

    diverse_scores = bald._apply_diversity(candidates, scores)
    assert len(diverse_scores) == 20
    # Should have -inf for non-selected
    assert np.sum(np.isneginf(diverse_scores)) == 15
    # Should have finite scores for selected
    assert np.sum(np.isfinite(diverse_scores)) == 5


def test_bald():
    """Test BALD acquisition function."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    bald = BALD(config)

    candidates = jnp.ones((10, 5))

    # Test with ensemble_params
    ensemble_params = [params for _ in range(3)]
    scores = bald.score(model, params, candidates, ensemble_params=ensemble_params)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
    print(f"BALD scores (ensemble): {scores}")

    # Test with MC dropout
    scores = bald.score(model, params, candidates, n_mc_samples=10)
    assert scores.shape == (10,)
    print(f"BALD scores (dropout): {scores}")


def test_expected_improvement():
    """Test Expected Improvement acquisition."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    ei = ExpectedImprovement(config, xi=0.01)

    candidates = jnp.ones((10, 5))
    ensemble_params = [params for _ in range(3)]
    y_best = 0.5

    scores = ei.score(model, params, candidates, y_best=y_best, ensemble_params=ensemble_params)
    assert scores.shape == (10,)
    assert np.all(scores >= 0)
    print(f"EI scores: {scores}")


def test_max_variance():
    """Test MaxVariance acquisition."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    mv = MaxVariance(config)
    candidates = jnp.ones((10, 5))

    scores = mv.score(model, params, candidates)
    assert scores.shape == (10,)
    assert np.all(scores >= 0)
    print(f"MaxVariance scores: {scores}")


def test_max_entropy():
    """Test MaxEntropy acquisition."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    me = MaxEntropy(config)
    candidates = jnp.ones((10, 5))

    scores = me.score(model, params, candidates)
    assert scores.shape == (10,)
    print(f"MaxEntropy scores: {scores}")


def test_diversity_acquisition():
    """Test DiversityAcquisition."""
    config = NeuralDiscoveryConfig()
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    div = DiversityAcquisition(config)
    candidates = jnp.ones((10, 5))

    # First selection
    scores1 = div.score(model, params, candidates)
    assert scores1.shape == (10,)

    # With already selected
    already_selected = np.array([0, 2])
    scores2 = div.score(model, params, candidates, already_selected=already_selected)
    assert scores2.shape == (10,)
    print(f"Diversity scores: {scores2}")


def test_acquisition_factory():
    """Test create_acquisition_function factory."""
    config = NeuralDiscoveryConfig()

    for acq_type in AcquisitionType.__args__:  # type: ignore
        acq_fn = create_acquisition_function(config, acq_type)
        assert acq_fn is not None
        print(f"Created {acq_type}: {type(acq_fn).__name__}")

    # Invalid type
    with pytest.raises(ValueError):
        create_acquisition_function(config, "invalid")  # type: ignore[arg-type]


def test_acquisition_select():
    """Test AcquisitionFunction.select method."""
    config = NeuralDiscoveryConfig(acquisition_batch_size=3)
    model = MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (10, 5), key, config)

    bald = BALD(config)
    candidates = jnp.ones((10, 5))
    ensemble_params = [params for _ in range(3)]

    result = bald.select(model, params, candidates, n_select=3, ensemble_params=ensemble_params)

    assert len(result.indices) == 3
    assert len(result.scores) == 3
    assert result.metadata['n_candidates'] == 10
    assert result.metadata['n_selected'] == 3
    print(f"Selected indices: {result.indices}")


def test_active_learning_loop():
    """Test ActiveLearningLoop (smoke test)."""
    config = NeuralDiscoveryConfig(
        max_epochs=2,
        acquisition_batch_size=2,
    )

    def model_fn():
        return MLPResNet(hidden_dims=(16,), num_layers=1, output_dim=1)

    bald = BALD(config)
    loop = ActiveLearningLoop(model_fn, config, bald)

    # Initial labeled data
    x_labeled = np.random.randn(10, 5)
    y_labeled = x_labeled[:, 0] + 0.1 * np.random.randn(10)

    # Unlabeled pool
    pool = np.random.randn(20, 5)

    # Labeling function (simulate ground truth)
    def labeling_fn(x):
        return x[:, 0] + 0.1 * np.random.randn(len(x))

    key = jax.random.PRNGKey(42)
    result = loop.run(
        (x_labeled, y_labeled),
        pool,
        labeling_fn,
        n_rounds=2,
        key=key,
    )

    assert 'final_labeled' in result
    assert 'final_pool' in result
    assert 'history' in result
    assert len(result['history']) == 2
    print(f"Final labeled: {len(result['final_labeled'][0])}, Pool: {len(result['final_pool'])}")


if __name__ == "__main__":
    test_acquisition_function_base()
    test_bald()
    test_expected_improvement()
    test_max_variance()
    test_max_entropy()
    test_diversity_acquisition()
    test_acquisition_factory()
    test_acquisition_select()
    test_active_learning_loop()
    print("All active learning tests passed!")