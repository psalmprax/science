# Databricks notebook source
"""Integration between neural discovery and the existing symbolic pipeline."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from flax import linen as nn
from sklearn.model_selection import train_test_split

from apedp.discovery import sparse_select_terms
from apedp.neural_discovery.architectures import create_model
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.neural_discovery.training import NeuralDiscoveryTrainer, TrainingHistory
from apedp.neural_discovery.uncertainty import DeepEnsemble, predict_with_uncertainty

Array = jax.Array


def _matrix(x_data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = list(x_data)
    if not names:
        raise ValueError("at least one input variable is required")
    columns = [np.asarray(x_data[name], dtype=np.float32).ravel() for name in names]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("all input variables must have the same length")
    return np.column_stack(columns), names


@dataclass
class NeuralDiscoveryResult:
    model: nn.Module
    params: dict[str, Any]
    trainer: NeuralDiscoveryTrainer
    history: TrainingHistory
    uncertainty_engine: DeepEnsemble | None = None
    symbolic_expressions: list[str] | None = None


class NeuralDiscoveryEngine:
    """Train neural surrogates and expose uncertainty-aware discovery helpers."""

    def __init__(self, config: NeuralDiscoveryConfig | None = None):
        self.config = config or NeuralDiscoveryConfig()
        self._trainer = NeuralDiscoveryTrainer(self.config)

    def fit(self, x: Array, y: Array, val_split: float = 0.2,
            key: Array | None = None, verbose: bool = True) -> NeuralDiscoveryResult:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if len(x) != len(y) or len(x) < 4:
            raise ValueError("x and y need equal lengths and at least four samples")
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=val_split, random_state=self.config.seed)
        model = create_model(self.config, input_dim=x.shape[1], output_dim=1)
        params, history = self._trainer.train(model, (train_x, train_y), (val_x, val_y), key if key is not None else jax.random.PRNGKey(self.config.seed), verbose=verbose)
        ensemble = None
        if self.config.uncertainty_method == "ensemble":
            ensemble = DeepEnsemble(lambda: create_model(self.config, input_dim=x.shape[1], output_dim=1), self.config)
            ensemble.train_ensemble((train_x, train_y), (val_x, val_y), key if key is not None else jax.random.PRNGKey(self.config.seed))
        return NeuralDiscoveryResult(model, params, self._trainer, history, ensemble)

    def predict_with_uncertainty(self, result: NeuralDiscoveryResult, x: Array) -> dict[str, Array]:
        if result.uncertainty_engine and result.uncertainty_engine.models:
            output = result.uncertainty_engine.predict(np.asarray(x, dtype=np.float32))
        else:
            output = predict_with_uncertainty(result.model, result.params, np.asarray(x, dtype=np.float32), method="dropout", n_samples=self.config.dropout_mc_samples)
        return {"mean": output.mean, "std": output.std, "lower": output.lower, "upper": output.upper}

    def neural_relevance(self, x_data: dict[str, np.ndarray], y_data: np.ndarray, seed: int = 42) -> dict:
        x, names = _matrix(x_data)
        y = np.asarray(y_data, dtype=np.float32).ravel()
        if len(y) < 30:
            return {"relevant": [], "note": "insufficient data for neural surrogate"}
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=seed)
        result = self.fit(train_x, train_y, key=jax.random.PRNGKey(seed), verbose=False)
        predictions = np.asarray(result.model.apply({"params": result.params}, test_x)).ravel()
        total = np.sum((test_y - np.mean(test_y)) ** 2) + 1e-12
        r2 = 1.0 - float(np.sum((test_y - predictions) ** 2)) / float(total)
        if r2 < 0.5:
            return {"surrogate_r2": r2, "relevant": [], "note": "held-out surrogate too weak"}
        baseline = float(np.mean((test_y - predictions) ** 2))
        rng = np.random.default_rng(seed)
        importance = {}
        for index, name in enumerate(names):
            permuted = test_x.copy()
            permuted[:, index] = rng.permutation(permuted[:, index])
            permuted_pred = np.asarray(result.model.apply({"params": result.params}, permuted)).ravel()
            importance[name] = float(np.mean((test_y - permuted_pred) ** 2) - baseline)
        floor = max(0.01, 0.05 * max(importance.values(), default=0.0))
        return {"surrogate_r2": round(r2, 4), "relevant": sorted((n for n, score in importance.items() if score > floor), key=importance.get, reverse=True), "relevance": importance, "note": "held-out permutation importance"}

    def distill_via_surrogate(self, x_data: dict[str, np.ndarray], y_data: np.ndarray, seed: int = 42, n_dense: int = 200) -> dict:
        x, names = _matrix(x_data)
        y = np.asarray(y_data, dtype=np.float32).ravel()
        if len(y) < 30:
            return {"terms": [], "note": "insufficient data for distillation"}
        result = self.fit(x, y, key=jax.random.PRNGKey(seed), verbose=False)
        rng = np.random.default_rng(seed)
        grid = x[rng.integers(0, len(x), size=n_dense)]
        predictions = np.asarray(result.model.apply({"params": result.params}, grid)).ravel()
        selected = sparse_select_terms({name: grid[:, i] for i, name in enumerate(names)}, predictions, selector="bic_forward", strict=False)
        return {"terms": selected.get("terms", []), "surrogate_r2": result.history.val_r2[-1] if result.history.val_r2 else float("nan"), "converged": bool(result.history.val_loss), "n_iter": len(result.history.train_loss), "method": "trained_surrogate_bic_distillation"}

    def discover_symbolic(self, x_data: dict[str, np.ndarray], y_data: np.ndarray, **_: Any) -> dict:
        x, names = _matrix(x_data)
        selected = sparse_select_terms({name: x[:, i] for i, name in enumerate(names)}, np.asarray(y_data, dtype=float).ravel(), selector="bic_forward", strict=False)
        expressions = selected.get("terms", [])
        return {"params": selected, "expressions": expressions, "valid_rate": 1.0 if expressions else 0.0}


@dataclass
class ModelDiscoveryIntegration:
    config: NeuralDiscoveryConfig = field(default_factory=NeuralDiscoveryConfig)
    engine: NeuralDiscoveryEngine = field(init=False)

    def __post_init__(self) -> None:
        self.engine = NeuralDiscoveryEngine(self.config)

    def enhance_discover(self, base_discover_fn: Callable, x_data: dict[str, np.ndarray], y_data: np.ndarray) -> dict:
        result = dict(base_discover_fn(x_data, y_data))
        relevance = self.engine.neural_relevance(x_data, y_data)
        distilled = self.engine.distill_via_surrogate(x_data, y_data)
        result["neural"] = {"relevance": relevance.get("relevance", {}), "surrogate_r2": relevance.get("surrogate_r2", 0.0), "distilled_terms": distilled.get("terms", [])}
        return result

    def suggest_next_experiment(self, x_labeled: dict[str, np.ndarray], y_labeled: np.ndarray, x_pool: dict[str, np.ndarray], n_select: int = 5) -> dict[str, Any]:
        """Select from an unlabeled pool using only the separately labeled data."""
        from apedp.neural_discovery.active_learning import create_acquisition_function
        labeled_matrix, names = _matrix(x_labeled)
        pool_matrix, pool_names = _matrix(x_pool)
        if names != pool_names:
            raise ValueError("labeled and pool variables must have identical names and order")
        if len(y_labeled) != len(labeled_matrix):
            raise ValueError("y_labeled must align with x_labeled, not x_pool")
        if len(pool_matrix) == 0:
            raise ValueError("x_pool must contain at least one candidate")
        result = self.engine.fit(labeled_matrix, y_labeled, verbose=False)
        acquisition = create_acquisition_function(self.config)
        selection = acquisition.select(result.model, result.params, pool_matrix, n_select=min(n_select, len(pool_matrix)))
        return {"indices": selection.indices, "scores": selection.scores, "acquisition_type": self.config.acquisition_type, "candidate_count": len(pool_matrix)}


def neural_relevance(x_data: dict[str, np.ndarray], y_data: np.ndarray, seed: int = 42) -> dict:
    return NeuralDiscoveryEngine().neural_relevance(x_data, y_data, seed)


def distill_via_surrogate(x_data: dict[str, np.ndarray], y_data: np.ndarray, seed: int = 42, n_dense: int = 200) -> dict:
    return NeuralDiscoveryEngine().distill_via_surrogate(x_data, y_data, seed, n_dense)


__all__ = ["ModelDiscoveryIntegration", "NeuralDiscoveryEngine", "NeuralDiscoveryResult", "distill_via_surrogate", "neural_relevance"]