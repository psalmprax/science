"""Tests for neural discovery architectures."""

import jax
import jax.numpy as jnp

from apedp.neural_discovery.architectures import (
    count_params,
    create_model,
    init_model,
)
from apedp.neural_discovery.config import NeuralDiscoveryConfig


def test_mlp_resnet():
    """Test MLPResNet forward pass and parameter counting."""
    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(64, 32),
        num_layers=2,
    )
    model = create_model(config, input_dim=10, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (4, 10), key, config)

    x = jnp.ones((4, 10))
    out = model.apply({'params': params}, x)
    assert out.shape == (4, 1)

    n_params = count_params(params)
    assert n_params > 0
    print(f"MLPResNet params: {n_params}")


def test_symbolic_transformer():
    """Test SymbolicTransformer forward pass."""
    config = NeuralDiscoveryConfig(
        architecture="symbolic_transformer",
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        max_seq_len=32,
        vocab_size=50,
    )
    model = create_model(config, input_dim=10, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (4, 32), key, config)

    tokens = jnp.ones((4, 32), dtype=jnp.int32)
    out = model.apply({'params': params}, tokens)
    assert out.shape == (4, 1)

    n_params = count_params(params)
    print(f"SymbolicTransformer params: {n_params}")


def test_equation_gnn():
    """Test EquationGNN forward pass."""
    config = NeuralDiscoveryConfig(
        architecture="equation_gnn",
        gnn_hidden_dim=32,
        gnn_num_layers=2,
    )
    model = create_model(config, input_dim=10, output_dim=1)
    key = jax.random.PRNGKey(42)

    # GNN needs different input format
    node_features = jnp.ones((8, 16))
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]])
    node_types = jnp.zeros(8, dtype=jnp.int32)

    params = model.init(key, node_features, edges, node_types, train=False)['params']

    out = model.apply({'params': params}, node_features, edges, node_types)
    assert out.shape == (1,)

    n_params = count_params(params)
    print(f"EquationGNN params: {n_params}")


def test_pinn():
    """Test PhysicsInformedNN forward pass."""
    config = NeuralDiscoveryConfig(
        architecture="pinn",
        hidden_dims=(64, 32),
        num_layers=2,
    )
    model = create_model(config, input_dim=3, output_dim=1)
    key = jax.random.PRNGKey(42)
    params = init_model(model, (4, 3), key, config)

    coords = jnp.ones((4, 3))
    out = model.apply({'params': params}, coords)
    assert out.shape == (4, 1)

    n_params = count_params(params)
    print(f"PINN params: {n_params}")


def test_all_architectures():
    """Test all architectures can be created and initialized."""
    configs = [
        NeuralDiscoveryConfig(architecture="mlp_resnet"),
        NeuralDiscoveryConfig(architecture="symbolic_transformer"),
        NeuralDiscoveryConfig(architecture="equation_gnn"),
        NeuralDiscoveryConfig(architecture="pinn"),
    ]

    for config in configs:
        model = create_model(config, input_dim=10, output_dim=1)
        key = jax.random.PRNGKey(42)

        if config.architecture == "equation_gnn":
            node_features = jnp.ones((8, 16))
            edges = jnp.array([[0, 1], [1, 2]])
            node_types = jnp.zeros(8, dtype=jnp.int32)
            params = model.init(key, node_features, edges, node_types, train=False)['params']
        elif config.architecture == "symbolic_transformer":
            tokens = jnp.ones((4, 32), dtype=jnp.int32)
            params = model.init(key, tokens, train=False)['params']
        else:
            params = init_model(model, (4, 10), key, config)

        n_params = count_params(params)
        assert n_params > 0
        print(f"{config.architecture}: {n_params} params")

    print("All architectures OK!")


if __name__ == "__main__":
    test_mlp_resnet()
    test_symbolic_transformer()
    test_equation_gnn()
    test_pinn()
    test_all_architectures()
    print("All architecture tests passed!")