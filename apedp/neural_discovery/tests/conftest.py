"""Shared test configuration and fixtures for neural discovery tests."""

import jax
import numpy as np
import pytest

# Configure JAX for testing
jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope="session")
def rng_key():
    """Provide a fixed PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def synthetic_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(42)
    n = 100
    d = 5
    x = np.random.randn(n, d).astype(np.float32)
    # y = 2*x0 - x1 + 0.5*x2^2 + noise
    y = (2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2]**2 + 0.1 * np.random.randn(n)).astype(np.float32)
    return x, y


@pytest.fixture(scope="session")
def synthetic_data_dict():
    """Generate synthetic data as dict for symbolic regression."""
    np.random.seed(42)
    n = 50
    x0 = np.random.uniform(1, 5, n).astype(np.float32)
    x1 = np.random.uniform(1, 5, n).astype(np.float32)
    y = (x0 + x1 + 0.1 * np.random.randn(n)).astype(np.float32)
    return {'x0': x0, 'x1': x1}, y


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(42)