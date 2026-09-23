# Databricks notebook source
"""Uncertainty quantification for neural discovery.

Implements:
- BayesianNN: Mean-field variational inference (SVI)
- DeepEnsemble: Multiple independently trained members
- ConformalPredictor: Split conformal prediction for distribution-free coverage
- CalibrationMetrics: ECE, NLL, sharpness, coverage
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import jit, random

from apedp.neural_discovery.architectures import init_model
from apedp.neural_discovery.config import NeuralDiscoveryConfig

Array = jnp.ndarray
PRNGKey = jax.Array
Variables = dict[str, Array]


@dataclass
class UncertaintyOutput:
    """Container for predictions with uncertainty."""
    mean: Array
    std: Array
    lower: Array  # Lower confidence bound
    upper: Array  # Upper confidence bound
    epistemic: Array | None = None  # Model uncertainty
    aleatoric: Array | None = None  # Data uncertainty


class UncertaintyWrapper(nn.Module):
    """Wrapper that adds uncertainty quantification to any model."""

    base_model: nn.Module
    method: str = "ensemble"  # ensemble, dropout, svi
    num_samples: int = 30

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> UncertaintyOutput:
        if self.method == "ensemble":
            return self._ensemble_forward(x, train)
        elif self.method == "dropout":
            return self._dropout_forward(x, train)
        elif self.method == "svi":
            return self._svi_forward(x, train)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _ensemble_forward(self, x: Array, train: bool) -> UncertaintyOutput:
        # Ensemble: multiple forward passes with different params
        # Note: In practice, we'd have multiple parameter sets
        # Here we simulate with dropout at test time as approximation
        preds = []
        for sample_index in range(self.num_samples):
            pred = self.base_model.apply(
                {'params': self.variables['params']}, x, train=True,
                rngs={'dropout': random.PRNGKey(sample_index)},
            )
            preds.append(pred)
        preds = jnp.stack(preds, axis=0)  # (num_samples, batch, output_dim)
        mean = jnp.mean(preds, axis=0)
        std = jnp.std(preds, axis=0)
        lower = mean - 1.645 * std  # 90% CI
        upper = mean + 1.645 * std
        return UncertaintyOutput(mean=mean, std=std, lower=lower, upper=upper)

    def _dropout_forward(self, x: Array, train: bool) -> UncertaintyOutput:
        # MC Dropout: multiple forward passes with dropout at test time
        preds = []
        for sample_index in range(self.num_samples):
            pred = self.base_model.apply(
                {'params': self.variables['params']}, x, train=True,
                rngs={'dropout': random.PRNGKey(sample_index)},
            )
            preds.append(pred)
        preds = jnp.stack(preds, axis=0)
        mean = jnp.mean(preds, axis=0)
        std = jnp.std(preds, axis=0)
        lower = mean - 1.645 * std
        upper = mean + 1.645 * std
        return UncertaintyOutput(mean=mean, std=std, lower=lower, upper=upper)

    def _svi_forward(self, x: Array, train: bool) -> UncertaintyOutput:
        """Use the fitted Bayesian parameter posterior for prediction."""
        posterior = BayesianNN(self.base_model, NeuralDiscoveryConfig())
        variational = {"variational": self.variables["variational"]}
        samples = posterior.sample_params(
            variational, self.make_rng("sample"), self.num_samples
        )["params"]
        predictions = []
        for index in range(self.num_samples):
            predictions.append(self.base_model.apply({"params": jax.tree_util.tree_map(
                lambda value, index=index: value[index], samples
            )}, x))
        predictions = jnp.stack(predictions)
        mean = jnp.mean(predictions, axis=0)
        std = jnp.std(predictions, axis=0)
        return UncertaintyOutput(mean, std, mean - 1.96 * std, mean + 1.96 * std, epistemic=std)


class BayesianNN:
    """Mean-field Variational Inference for Bayesian Neural Networks.

    Uses BlackJAX for SVI with reparameterization trick.
    """

    def __init__(self,
                 model: nn.Module,
                 config: NeuralDiscoveryConfig,
                 prior_std: float = 1.0):
        self.model = model
        self.config = config
        self.prior_std = prior_std
        self.param_shapes = None
        self.param_names = None
        self._variational_template = None
        self._base_params = None

    def init_variational_params(self, key: PRNGKey, input_shape: tuple[int, ...]) -> Variables:
        """Initialize variational parameters (mean and log_std for each weight)."""
        # Get base model param shapes
        dummy_input = jnp.ones(input_shape)
        base_params = self.model.init(key, dummy_input)['params']
        self._base_params = base_params
        self.param_shapes = jax.tree_util.tree_map(jnp.shape, base_params)
        self.param_names = list(jax.tree_util.tree_leaves_with_path(base_params))
        leaves = jax.tree_util.tree_leaves(base_params)
        var_params = {}
        for index, value in enumerate(leaves):
            var_params[f"param_{index}_mean"] = value
            var_params[f"param_{index}_log_std"] = jnp.full_like(value, -5.0)
        self._variational_template = var_params
        return {"variational": var_params}

    def sample_params(self, var_params: Variables, key: PRNGKey,
                      n_samples: int = 1) -> Variables:
        """Sample parameters from variational posterior using reparameterization."""
        variational = var_params["variational"]
        leaves = []
        keys = random.split(key, len(variational) // 2)
        for index, sample_key in enumerate(keys):
            mean = variational[f"param_{index}_mean"]
            log_std = variational[f"param_{index}_log_std"]
            leaves.append(mean + jnp.exp(log_std) * random.normal(
                sample_key, (n_samples,) + mean.shape
            ))
        samples = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(self._base_params), leaves
        )
        return {"params": samples}

    def fit_variational(self, train_ds: tuple[Array, Array], key: PRNGKey | None = None,
                        steps: int = 100, learning_rate: float = 1e-3,
                        n_samples: int = 3) -> dict[str, Any]:
        """Optimize the mean-field ELBO with reparameterized gradients."""
        x, y = train_ds
        if key is None:
            key = random.PRNGKey(self.config.seed)
        if self._variational_template is None:
            params = self.init_variational_params(key, tuple(x.shape))
        else:
            params = {"variational": self._variational_template}
        optimizer = optax.adam(learning_rate)
        state = optimizer.init(params)

        def objective(current, sample_key):
            loss, _ = self.elbo_loss(current, (x, y), sample_key, n_samples)
            return jnp.asarray(loss)

        history = []
        for step in range(steps):
            step_key = random.fold_in(key, step)
            loss, grads = jax.value_and_grad(objective)(params, step_key)
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            history.append(float(loss))
        self._variational_template = params["variational"]
        return {"variational": params, "loss": history[-1], "loss_history": history}

    def elbo_loss(self, var_params: Variables, batch: tuple[Array, Array],
                  key: PRNGKey, n_samples: int = 5) -> tuple[float, dict]:
        """Compute ELBO: E[log p(y|x,w)] - KL[q(w)||p(w)].

        Args:
            var_params: Variational parameters
            batch: (x, y) training batch
            key: PRNG key
            n_samples: Number of MC samples for expectation

        Returns:
            ELBO loss (negative for minimization), info dict
        """
        x, y = batch
        batch_size = x.shape[0]

        # Sample weights
        total_log_lik = jnp.asarray(0.0)

        for i in range(n_samples):
            sample_key = random.fold_in(key, i)
            sampled_params = self.sample_params(var_params, sample_key, n_samples=1)
            # Squeeze sample dimension
            sampled_params = jax.tree_util.tree_map(lambda x: x[0], sampled_params['params'])

            # Predictions
            preds = self.model.apply({'params': sampled_params}, x)

            # Log likelihood (Gaussian)
            log_lik = -0.5 * jnp.sum((preds - y) ** 2) - 0.5 * batch_size * jnp.log(2 * jnp.pi)
            total_log_lik += log_lik

        variational = var_params["variational"]
        total_kl = sum(
            jnp.sum(variational[f"param_{index}_log_std"] - jnp.log(self.prior_std) +
                    (jnp.exp(2 * variational[f"param_{index}_log_std"]) +
                     variational[f"param_{index}_mean"]**2) /
                    (2 * self.prior_std**2) - 0.5)
            for index in range(len(variational) // 2)
        )

        elbo = -(total_log_lik / n_samples - total_kl / n_samples)
        return elbo, {
            'log_lik': total_log_lik / n_samples,
            'kl': total_kl / n_samples,
        }


class DeepEnsemble:
    """Deep Ensemble: Multiple independently trained models.

    Provides epistemic uncertainty through ensemble disagreement.
    """

    def __init__(self,
                 model_fn: Callable[[], nn.Module],
                 config: NeuralDiscoveryConfig):
        self.model_fn = model_fn
        self.config = config
        self.ensemble_size = config.ensemble_size
        self.models = []
        self.params_list = []

    def train_ensemble(self,
                       train_ds: tuple[Array, Array],
                       val_ds: tuple[Array, Array],
                       key: PRNGKey) -> dict[str, Any]:
        """Train ensemble members with different seeds.

        Args:
            train_ds: (x_train, y_train)
            val_ds: (x_val, y_val)
            key: PRNG key

        Returns:
            Training history for each member
        """
        x_train, _y_train = train_ds
        _x_val, _y_val = val_ds
        histories = []

        for i in range(self.ensemble_size):
            member_key = random.fold_in(key, i)
            model = self.model_fn()
            params = init_model(model, x_train.shape, member_key, self.config)

            # Train single member (simplified - use actual trainer)
            history = self._train_member(model, params, train_ds, val_ds, member_key)
            self.models.append(model)
            self.params_list.append(history['best_params'])
            histories.append(history)

        return {'histories': histories}

    def _train_member(self, model: nn.Module, params: Variables,
                      train_ds: tuple[Array, Array], val_ds: tuple[Array, Array],
                      key: PRNGKey) -> dict:
        """Train a single ensemble member."""
        # Simplified training loop - in practice use NeuralDiscoveryTrainer
        x_train, y_train = train_ds
        x_val, y_val = val_ds

        optimizer = optax.adam(self.config.learning_rate)
        opt_state = optimizer.init(params)

        best_val_loss = float('inf')
        best_params = params
        patience = 0

        @jit
        def loss_fn(p, x, y):
            preds = model.apply({'params': p}, x)
            return jnp.mean((preds - y) ** 2)

        @jit
        def train_step(p, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, p)
            p = optax.apply_updates(p, updates)
            return p, opt_state, loss

        for epoch in range(self.config.max_epochs):
            # Shuffle
            perm = random.permutation(key, len(x_train))
            x_shuffled = x_train[perm]
            y_shuffled = y_train[perm]

            # Mini-batch training
            for i in range(0, len(x_train), self.config.batch_size):
                x_batch = x_shuffled[i:i + self.config.batch_size]
                y_batch = y_shuffled[i:i + self.config.batch_size]
                params, opt_state, _ = train_step(params, opt_state, x_batch, y_batch)

            # Validation
            val_preds = model.apply({'params': params}, x_val)
            val_loss = float(jnp.mean((val_preds - y_val) ** 2))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break

        return {'best_params': best_params, 'best_val_loss': best_val_loss}

    def predict(self, x: Array) -> UncertaintyOutput:
        """Predict with ensemble uncertainty."""
        preds = []
        for model, params in zip(self.models, self.params_list):
            pred = model.apply({'params': params}, x)
            preds.append(pred)
        preds = jnp.stack(preds, axis=0)  # (ensemble_size, batch, output_dim)

        mean = jnp.mean(preds, axis=0)
        std = jnp.std(preds, axis=0)
        epistemic = std  # Epistemic uncertainty = ensemble disagreement

        # Aleatoric would need separate modeling
        lower = mean - 1.96 * std  # 95% CI
        upper = mean + 1.96 * std

        return UncertaintyOutput(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            epistemic=epistemic,
            aleatoric=jnp.zeros_like(std),  # Placeholder
        )

    def pairwise_disagreement(self, x: Array) -> float:
        """Compute average pairwise disagreement among ensemble members."""
        preds = []
        for model, params in zip(self.models, self.params_list):
            pred = model.apply({'params': params}, x)
            preds.append(pred)
        preds = jnp.stack(preds, axis=0)

        # Pairwise MSE
        n = len(preds)
        total_disagreement = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_disagreement += jnp.mean((preds[i] - preds[j]) ** 2)
                count += 1
        return float(total_disagreement / count) if count > 0 else 0.0


class ConformalPredictor:
    """Split Conformal Prediction for distribution-free uncertainty.

    Provides finite-sample coverage guarantees without distributional assumptions.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate (1 - coverage). Default 0.1 = 90% coverage.
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None

    def calibrate(self, model: nn.Module, params: Variables,
                  cal_x: Array, cal_y: Array) -> dict[str, float]:
        """Calibrate on hold-out calibration set.

        Args:
            model: Trained model
            params: Model parameters
            cal_x: Calibration inputs
            cal_y: Calibration targets

        Returns:
            Calibration info
        """
        preds = model.apply({'params': params}, cal_x)
        scores = jnp.abs(preds - cal_y)  # Nonconformity scores

        # Compute conformal quantile
        n = len(scores)
        q_level = min(1.0, float(jnp.ceil((n + 1) * (1 - self.alpha)) / n))
        self.quantile = float(jnp.quantile(scores, q_level))
        self.calibration_scores = scores

        # Empirical coverage
        coverage = float(jnp.mean(scores <= self.quantile))

        return {
            'quantile': self.quantile,
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'n_cal': n,
        }

    def predict(self, model: nn.Module, params: Variables, x: Array) -> UncertaintyOutput:
        """Predict with conformal prediction intervals."""
        if self.quantile is None:
            raise ValueError("Must call calibrate() first")

        preds = model.apply({'params': params}, x)
        lower = preds - self.quantile
        upper = preds + self.quantile

        return UncertaintyOutput(
            mean=preds,
            std=jnp.full_like(preds, self.quantile),
            lower=lower,
            upper=upper,
        )


class CalibrationMetrics:
    """Metrics for uncertainty calibration."""

    @staticmethod
    def expected_calibration_error(mean: Array, std: Array, y_true: Array,
                                   n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE) for regression.

        Bins predictions by predicted std, compares empirical coverage vs nominal.
        """
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        y_true = np.asarray(y_true).ravel()

        # Bin by predicted std
        bin_edges = np.percentile(std, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ece = 0.0
        for i in range(n_bins):
            mask = (std >= bin_edges[i]) & (std < bin_edges[i + 1])
            if np.sum(mask) == 0:
                continue

            bin_mean = mean[mask]
            bin_std = std[mask]
            bin_y = y_true[mask]

            # Nominal coverage for 1-sigma
            nominal = 0.6827
            empirical = np.mean(np.abs(bin_y - bin_mean) <= bin_std)
            ece += np.abs(empirical - nominal) * np.sum(mask)

        return float(ece / len(y_true))

    @staticmethod
    def coverage_at_level(mean: Array, std: Array, y_true: Array,
                          levels: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
        """Compute empirical coverage at multiple confidence levels."""
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        y_true = np.asarray(y_true).ravel()

        results = {}
        for level in levels:
            z = {0.5: 0.674, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
            lower = mean - z * std
            upper = mean + z * std
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            results[f'coverage_{int(level*100)}'] = float(coverage)
            results[f'target_{int(level*100)}'] = level
        return results

    @staticmethod
    def negative_log_likelihood(mean: Array, std: Array, y_true: Array) -> float:
        """Gaussian NLL."""
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        y_true = np.asarray(y_true).ravel()
        nll = 0.5 * np.log(2 * np.pi * std**2) + 0.5 * (y_true - mean)**2 / std**2
        return float(np.mean(nll))

    @staticmethod
    def sharpness(std: Array) -> float:
        """Mean prediction interval width (sharpness)."""
        return float(np.mean(std))

    @staticmethod
    def calibration_curve(mean: Array, std: Array, y_true: Array,
                          n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Reliability diagram data: (nominal_coverage, empirical_coverage)."""
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        y_true = np.asarray(y_true).ravel()

        nominal_levels = np.linspace(0.05, 0.95, n_bins)
        empirical = []

        for level in nominal_levels:
            z = np.sqrt(2) * scipy.special.erfinv(level)
            lower = mean - z * std
            upper = mean + z * std
            cov = np.mean((y_true >= lower) & (y_true <= upper))
            empirical.append(cov)

        return nominal_levels, np.array(empirical)


def predict_with_uncertainty(model: nn.Module,
                             params: Variables,
                             x: Array,
                             method: str = "ensemble",
                             ensemble_params: list | None = None,
                             conformal: ConformalPredictor | None = None,
                             n_samples: int = 30) -> UncertaintyOutput:
    """Unified prediction with uncertainty.

    Args:
        model: Flax model
        params: Model parameters (single or ensemble)
        x: Input data
        method: "ensemble", "dropout", "conformal"
        ensemble_params: List of param dicts for ensemble
        conformal: Calibrated ConformalPredictor
        n_samples: MC samples for dropout

    Returns:
        UncertaintyOutput with mean, std, lower, upper
    """
    if method == "ensemble" and ensemble_params:
        preds = [model.apply({'params': p}, x) for p in ensemble_params]
        preds = jnp.stack(preds, axis=0)
        mean = jnp.mean(preds, axis=0)
        std = jnp.std(preds, axis=0)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        return UncertaintyOutput(mean=mean, std=std, lower=lower, upper=upper,
                                 epistemic=std)

    elif method == "dropout":
        preds = []
        for sample_index in range(n_samples):
            pred = model.apply(
                {'params': params},
                x,
                train=True,
                rngs={'dropout': random.PRNGKey(sample_index)},
            )
            preds.append(pred)
        preds = jnp.stack(preds, axis=0)
        mean = jnp.mean(preds, axis=0)
        std = jnp.std(preds, axis=0)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        return UncertaintyOutput(mean=mean, std=std, lower=lower, upper=upper)

    elif method == "conformal" and conformal:
        return conformal.predict(model, params, x)

    else:
        # Single point prediction
        pred = model.apply({'params': params}, x)
        return UncertaintyOutput(
            mean=pred, std=jnp.zeros_like(pred),
            lower=pred, upper=pred
        )


# Try to import scipy for calibration curve
try:
    import scipy.special
except ImportError:
    scipy = None

__all__ = [
    "BayesianNN",
    "CalibrationMetrics",
    "ConformalPredictor",
    "DeepEnsemble",
    "UncertaintyOutput",
    "UncertaintyWrapper",
    "predict_with_uncertainty",
]

# COMMAND ----------

# Aleatoric uncertainty is not modeled by the current deterministic regressors.
# Do not expose a fabricated zero-valued estimate as measured data uncertainty.
_original_deep_ensemble_predict = DeepEnsemble.predict


def _deep_ensemble_predict_without_fake_aleatoric(self, x):
    output = _original_deep_ensemble_predict(self, x)
    return UncertaintyOutput(
        mean=output.mean,
        std=output.std,
        lower=output.lower,
        upper=output.upper,
        epistemic=output.epistemic,
        aleatoric=None,
    )


DeepEnsemble.predict = _deep_ensemble_predict_without_fake_aleatoric
