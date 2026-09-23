"""Tests for neural discovery config."""

import pytest

from apedp.neural_discovery.config import (
    PRESET_CONFIGS,
    NeuralDiscoveryConfig,
    TrainingConfig,
    get_default_config,
    get_preset,
)


def test_default_config():
    """Test default configuration creation."""
    config = get_default_config()
    assert isinstance(config, NeuralDiscoveryConfig)
    assert config.architecture == "mlp_resnet"
    assert config.ensemble_size == 5
    assert config.cv_folds == 5


def test_config_validation():
    """Test config validation."""
    # Valid config
    config = NeuralDiscoveryConfig(ensemble_size=5, cv_folds=5)
    assert config.ensemble_size == 5

    # Invalid ensemble_size
    with pytest.raises(ValueError):
        NeuralDiscoveryConfig(ensemble_size=1)

    # Invalid conformal_alpha
    with pytest.raises(ValueError):
        NeuralDiscoveryConfig(conformal_alpha=1.5)

    # Invalid cv_folds
    with pytest.raises(ValueError):
        NeuralDiscoveryConfig(cv_folds=1)

    # Invalid max_epochs
    with pytest.raises(ValueError):
        NeuralDiscoveryConfig(max_epochs=0)


def test_config_with_updates():
    """Test creating modified configs."""
    base = NeuralDiscoveryConfig(learning_rate=1e-3, max_epochs=100)
    updated = base.with_updates(learning_rate=1e-4, max_epochs=200)

    assert updated.learning_rate == 1e-4
    assert updated.max_epochs == 200
    # Other fields unchanged
    assert updated.ensemble_size == base.ensemble_size


def test_training_config():
    """Test TrainingConfig creation."""
    config = NeuralDiscoveryConfig(learning_rate=1e-3, weight_decay=1e-4)
    train_config = TrainingConfig.from_discovery_config(config)

    assert train_config.learning_rate == 1e-3
    assert train_config.weight_decay == 1e-4
    assert train_config.batch_size == config.batch_size


def test_presets():
    """Test all preset configurations."""
    for name in PRESET_CONFIGS:
        config = get_preset(name)
        assert isinstance(config, NeuralDiscoveryConfig)
        assert config.architecture in ["mlp_resnet", "symbolic_transformer", "equation_gnn", "pinn"]
        print(f"Preset {name}: {config.architecture}")

    # Unknown preset
    with pytest.raises(ValueError):
        get_preset("nonexistent")


def test_preset_values():
    """Test specific preset values."""
    fast = get_preset("fast")
    assert fast.max_epochs == 100
    assert fast.ensemble_size == 3
    assert fast.cv_folds == 3

    accurate = get_preset("accurate")
    assert accurate.max_epochs == 1000
    assert accurate.ensemble_size == 10
    assert accurate.hyperopt_trials == 100

    symbolic = get_preset("symbolic")
    assert symbolic.architecture == "symbolic_transformer"
    assert symbolic.symbolic_rl_epochs == 200

    physics = get_preset("physics")
    assert physics.architecture == "pinn"

    production = get_preset("production")
    assert production.uncertainty_method == "ensemble"


if __name__ == "__main__":
    test_default_config()
    test_config_validation()
    test_config_with_updates()
    test_training_config()
    test_presets()
    test_preset_values()
    print("All config tests passed!")