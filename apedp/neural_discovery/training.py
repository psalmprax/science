# Databricks notebook source
"""Training, Hyperparameter Optimization, and Cross-Validation for Neural Discovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import jit, random, value_and_grad
from sklearn.model_selection import KFold

from apedp.neural_discovery.architectures import init_model
from apedp.neural_discovery.config import NeuralDiscoveryConfig, TrainingConfig

Array = jnp.ndarray
PRNGKey = jax.Array
Variables = dict[str, Array]


@dataclass
class TrainState:
    """Training state (similar to Flax TrainState but simplified)."""
    step: int
    params: Variables
    opt_state: optax.OptState
    best_params: Variables | None = None
    best_val_loss: float = float('inf')
    patience_counter: int = 0


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_r2: list[float] = field(default_factory=list)
    val_r2: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)


class NeuralDiscoveryTrainer:
    """Unified training loop with k-fold CV, early stopping, LR scheduling."""

    def __init__(self, config: NeuralDiscoveryConfig):
        self.config = config
        self.training_config = TrainingConfig.from_discovery_config(config)

    def train(self,
              model: nn.Module,
              train_ds: tuple[Array, Array],
              val_ds: tuple[Array, Array] | None = None,
              key: PRNGKey | None = None,
              verbose: bool = True) -> tuple[Variables, TrainingHistory]:
        """Train a single model.

        Args:
            model: Flax model
            train_ds: (x_train, y_train)
            val_ds: (x_val, y_val) or None for no validation
            key: PRNG key
            verbose: Print progress

        Returns:
            Best parameters, training history
        """
        if key is None:
            key = random.PRNGKey(self.config.seed)
        x_train, y_train = train_ds
        n_train = len(x_train)

        # Initialize
        params = init_model(model, x_train.shape, key, self.config)
        optimizer = self._create_optimizer()
        opt_state = optimizer.init(params)

        # LR schedule
        lr_schedule = self._create_lr_schedule()

        @jit
        def loss_fn(params, x, y):
            preds = model.apply({'params': params}, x)
            return jnp.mean((preds - y) ** 2)

        @jit
        def train_step(params, opt_state, x, y, lr):
            loss, grads = value_and_grad(loss_fn)(params, x, y)
            # Gradient clipping
            grads = jax.tree_util.tree_map(
                lambda g: jnp.clip(g, -self.training_config.gradient_clip,
                                   self.training_config.gradient_clip), grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        step = 0
        best_params = None
        best_val_loss = float('inf')
        patience_counter = 0
        history = TrainingHistory()

        # Training loop
        for epoch in range(self.training_config.max_epochs):
            epoch_key = random.fold_in(key, epoch)

            # Shuffle
            perm = random.permutation(epoch_key, n_train)
            x_shuffled = x_train[perm]
            y_shuffled = y_train[perm]

            # Mini-batch training
            epoch_losses = []
            for i in range(0, n_train, self.training_config.batch_size):
                x_batch = x_shuffled[i:i + self.training_config.batch_size]
                y_batch = y_shuffled[i:i + self.training_config.batch_size]

                lr = lr_schedule(step)
                params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, lr)
                step += 1
                epoch_losses.append(float(loss))

            # Validation
            train_loss = np.mean(epoch_losses)
            train_r2 = self._compute_r2(model, params, x_train, y_train)

            history.train_loss.append(train_loss)
            history.train_r2.append(train_r2)
            history.learning_rate.append(float(lr_schedule(step)))

            val_loss = train_loss
            val_r2 = train_r2

            if val_ds is not None:
                x_val, y_val = val_ds
                val_preds = model.apply({'params': params}, x_val)
                val_loss = float(jnp.mean((val_preds - y_val) ** 2))
                val_r2 = self._compute_r2(model, params, x_val, y_val)

                history.val_loss.append(val_loss)
                history.val_r2.append(val_r2)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.training_config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % self.training_config.log_interval == 0:
                print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}, train_r2={train_r2:.4f}, "
                      f"val_r2={val_r2:.4f}, lr={lr_schedule(step):.6f}")

        # Return best params
        final_params = best_params if best_params is not None else params
        return final_params, history

    def cross_validate(self,
                       model_fn: Callable[[], nn.Module],
                       x: Array,
                       y: Array,
                       key: PRNGKey | None = None) -> dict[str, Any]:
        """k-fold cross-validation with repeated seeds.

        Args:
            model_fn: Function returning fresh model instance
            x: Features (n_samples, n_features)
            y: Targets (n_samples,)
            key: PRNG key

        Returns:
            CV results with mean/std across folds and seeds
        """
        if key is None:
            key = random.PRNGKey(self.config.seed)
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)

        all_fold_results = []

        for seed_offset in range(self.config.cv_seeds):
            fold_key = random.fold_in(key, seed_offset)
            fold_results = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x)):
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = model_fn()
                fold_key, subkey = random.split(fold_key)
                params, history = self.train(model, (x_train, y_train), (x_val, y_val), subkey, verbose=False)

                # Final evaluation
                val_preds = model.apply({'params': params}, x_val)
                val_loss = float(jnp.mean((val_preds - y_val) ** 2))
                val_r2 = self._compute_r2(model, params, x_val, y_val)

                fold_results.append({
                    'fold': fold_idx,
                    'seed': seed_offset,
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'train_loss': history.train_loss[-1],
                    'train_r2': history.train_r2[-1],
                    'epochs': len(history.train_loss),
                })

            all_fold_results.extend(fold_results)

        # Aggregate
        val_losses = [r['val_loss'] for r in all_fold_results]
        val_r2s = [r['val_r2'] for r in all_fold_results]

        return {
            'folds': all_fold_results,
            'val_loss_mean': float(np.mean(val_losses)),
            'val_loss_std': float(np.std(val_losses)),
            'val_r2_mean': float(np.mean(val_r2s)),
            'val_r2_std': float(np.std(val_r2s)),
            'n_folds': self.config.cv_folds,
            'n_seeds': self.config.cv_seeds,
        }

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with weight decay."""
        return optax.adamw(
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

    def _create_lr_schedule(self) -> Callable[[int], float]:
        """Create learning rate schedule."""
        if self.training_config.lr_schedule == "cosine":
            return optax.cosine_decay_schedule(
                init_value=self.training_config.learning_rate,
                decay_steps=self.training_config.max_epochs,
                alpha=0.01,
            )
        elif self.training_config.lr_schedule == "constant":
            return optax.constant_schedule(self.training_config.learning_rate)
        elif self.training_config.lr_schedule == "step":
            return optax.piecewise_constant_schedule(
                self.training_config.learning_rate,
                {self.training_config.max_epochs // 2: 0.1,
                 self.training_config.max_epochs * 3 // 4: 0.01}
            )
        else:
            return optax.constant_schedule(self.training_config.learning_rate)

    def _compute_r2(self, model: nn.Module, params: Variables, x: Array, y: Array) -> float:
        """Compute R² score."""
        preds = model.apply({'params': params}, x)
        ss_res = jnp.sum((y - preds) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-8))


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(self, config: NeuralDiscoveryConfig):
        self.config = config

    def optimize(self,
                 model_fn: Callable[[dict], nn.Module],
                 x: Array,
                 y: Array,
                 search_space: dict[str, Any],
                 key: PRNGKey | None = None) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            model_fn: Function taking config dict and returning model
            x: Features
            y: Targets
            search_space: Dict of parameter names to optuna distributions
            key: PRNG key

        Returns:
            Best parameters and optimization history
        """
        try:
            import optuna
        except ImportError as e:
            raise ImportError(
                "optuna required for hyperopt. Install with: pip install optuna"
            ) from e

        if key is None:
            key = random.PRNGKey(self.config.seed)

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for name, dist in search_space.items():
                params[name] = trial.suggest_float(name, *dist) if isinstance(dist, tuple) else dist

            # Create config with sampled params
            trial_config = self.config.with_updates(**params)
            NeuralDiscoveryTrainer(trial_config)  # construction validated below by CV

            # Quick CV (reduced folds/seeds for speed)
            quick_config = trial_config.with_updates(cv_folds=3, cv_seeds=1)
            quick_trainer = NeuralDiscoveryTrainer(quick_config)

            cv_result = quick_trainer.cross_validate(
                lambda: model_fn(params), x, y, key
            )

            return cv_result['val_loss_mean']

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner() if self.config.hyperopt_pruner == "median" else None,
        )

        study.optimize(objective, n_trials=self.config.hyperopt_trials,
                       timeout=self.config.hyperopt_timeout)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': [(t.params, t.value) for t in study.trials],
        }


class ModelCheckpoint:
    """Model checkpointing for best validation performance."""

    def __init__(self, save_dir: str, metric: str = "val_r2", mode: str = "max"):
        self.save_dir = save_dir
        self.metric = metric
        self.mode = mode
        self.best_value = -float('inf') if mode == "max" else float('inf')

    def save_if_best(self,
                     model: nn.Module,
                     params: Variables,
                     metrics: dict[str, float],
                     epoch: int,
                     step: int) -> bool:
        """Save checkpoint if metric improved."""
        import os
        os.makedirs(self.save_dir, exist_ok=True)

        value = metrics.get(self.metric, 0)
        improved = (value > self.best_value) if self.mode == "max" else (value < self.best_value)

        if improved:
            self.best_value = value
            path = f"{self.save_dir}/best_epoch_{epoch}_step_{step}.msgpack"
            self._save_params(params, path)
            return True
        return False

    def _save_params(self, params: Variables, path: str):
        """Save parameters using msgpack."""
        import msgpack
        import msgpack_numpy as m
        m.patch()
        # Convert JAX arrays to numpy for serialization
        params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
        with open(path, 'wb') as f:
            msgpack.pack(params_np, f)

    @staticmethod
    def load_params(path: str) -> Variables:
        """Load parameters from checkpoint."""
        import msgpack
        import msgpack_numpy as m
        m.patch()
        with open(path, 'rb') as f:
            return msgpack.unpack(f)


__all__ = [
    "HyperparameterOptimizer",
    "ModelCheckpoint",
    "NeuralDiscoveryTrainer",
    "TrainState",
    "TrainingHistory",
]