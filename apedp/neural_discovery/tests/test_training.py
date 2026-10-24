"""Tests for neural discovery training and hyperopt."""

import jax
import jax.numpy as jnp
import numpy as np

from apedp.neural_discovery.architectures import MLPResNet
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.neural_discovery.training import (
    ModelCheckpoint,
    NeuralDiscoveryTrainer,
    TrainingConfig,
)


def test_trainer_basic():
    """Test basic training loop."""
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(32, 16),
        num_layers=2,
        max_epochs=10,
        batch_size=8,
        early_stopping_patience=5,
    )

    trainer = NeuralDiscoveryTrainer(config)
    model = MLPResNet(hidden_dims=config.hidden_dims, num_layers=config.num_layers, output_dim=1)

    # Synthetic data
    np.random.seed(42)
    x = np.random.randn(100, 10).astype(np.float32)
    y = (x[:, 0] * 2 + x[:, 1] * -1 + 0.5 * np.random.randn(100)).astype(np.float32)

    x_train, y_train = x[:80], y[:80]
    x_val, y_val = x[80:], y[80:]

    key = jax.random.PRNGKey(42)
    params, history = trainer.train(model, (x_train, y_train), (x_val, y_val), key, verbose=False)

    assert params is not None
    assert len(history.train_loss) > 0
    assert len(history.val_loss) > 0
    print(f"Train loss: {history.train_loss[-1]:.4f}, Val loss: {history.val_loss[-1]:.4f}")
    print(f"Train R²: {history.train_r2[-1]:.4f}, Val R²: {history.val_r2[-1]:.4f}")


def test_cross_validate():
    """Test k-fold cross-validation."""
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(16,),
        num_layers=1,
        max_epochs=5,
        cv_folds=3,
        cv_seeds=1,
    )

    trainer = NeuralDiscoveryTrainer(config)

    def model_fn():
        return MLPResNet(hidden_dims=config.hidden_dims, num_layers=config.num_layers, output_dim=1)

    np.random.seed(42)
    x = np.random.randn(60, 5).astype(np.float32)
    y = (x[:, 0] + x[:, 1] * 0.5 + 0.1 * np.random.randn(60)).astype(np.float32)

    key = jax.random.PRNGKey(42)
    cv_result = trainer.cross_validate(model_fn, x, y, key)

    assert 'folds' in cv_result
    assert 'val_loss_mean' in cv_result
    assert 'val_loss_std' in cv_result
    assert 'val_r2_mean' in cv_result
    assert cv_result['n_folds'] == 3
    assert cv_result['n_seeds'] == 1
    print(f"CV val_loss: {cv_result['val_loss_mean']:.4f} ± {cv_result['val_loss_std']:.4f}")
    print(f"CV val_r2: {cv_result['val_r2_mean']:.4f} ± {cv_result['val_r2_std']:.4f}")


def test_lr_schedules():
    """Test different learning rate schedules."""
    for schedule in ["cosine", "constant", "step"]:
        config = NeuralDiscoveryConfig(
            lr_schedule=schedule,
            max_epochs=100,
            learning_rate=1e-3,
        )
        # Exercises the config translation; the trainer below is what this
        # test asserts on.
        _train_config = TrainingConfig.from_discovery_config(config)
        trainer = NeuralDiscoveryTrainer(config)

        # Test schedule function creation
        lr_fn = trainer._create_lr_schedule()
        lr_start = lr_fn(0)
        lr_mid = lr_fn(50)
        lr_end = lr_fn(99)

        assert lr_start > 0
        print(f"{schedule}: start={lr_start:.6f}, mid={lr_mid:.6f}, end={lr_end:.6f}")


def test_optimizer_creation():
    """Test optimizer creation with weight decay."""
    config = NeuralDiscoveryConfig(weight_decay=1e-4, learning_rate=1e-3)
    trainer = NeuralDiscoveryTrainer(config)
    optimizer = trainer._create_optimizer()

    # Test optimizer works
    params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros(5)}
    opt_state = optimizer.init(params)
    grads = {'w': jnp.ones((10, 5)) * 0.1, 'b': jnp.zeros(5) * 0.1}
    updates, _new_state = optimizer.update(grads, opt_state, params)
    import optax
    new_params = optax.apply_updates(params, updates)
    assert 'w' in new_params


def test_model_checkpoint():
    """Test ModelCheckpoint saving and loading."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = ModelCheckpoint(tmpdir, metric="val_r2", mode="max")

        # Mock params
        params = {'w': jnp.ones((5, 5)), 'b': jnp.zeros(5)}

        # First save (should save)
        saved = checkpoint.save_if_best(None, params, {'val_r2': 0.8}, epoch=0, step=100)
        assert saved is True

        # Better metric (should save)
        saved = checkpoint.save_if_best(None, params, {'val_r2': 0.9}, epoch=1, step=200)
        assert saved is True

        # Worse metric (should not save)
        saved = checkpoint.save_if_best(None, params, {'val_r2': 0.7}, epoch=2, step=300)
        assert saved is False

        # Check file exists
        files = os.listdir(tmpdir)
        assert len(files) >= 1
        print(f"Checkpoint files: {files}")


def test_training_config():
    """Test TrainingConfig dataclass."""
    config = NeuralDiscoveryConfig(
        learning_rate=5e-4,
        weight_decay=1e-5,
        batch_size=64,
        max_epochs=200,
    )
    train_config = TrainingConfig.from_discovery_config(config)

    assert train_config.learning_rate == 5e-4
    assert train_config.weight_decay == 1e-5
    assert train_config.batch_size == 64
    assert train_config.max_epochs == 200


if __name__ == "__main__":
    test_trainer_basic()
    test_cross_validate()
    test_lr_schedules()
    test_optimizer_creation()
    test_model_checkpoint()
    test_training_config()
    print("All training tests passed!")