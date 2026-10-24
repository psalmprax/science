# Databricks notebook source
"""Active Learning / Experimental Design for Neural Discovery.

Implements acquisition functions for optimal experiment design:
- BALD (Bayesian Active Learning by Disagreement)
- Expected Improvement (for optimization)
- MaxVariance / MaxEntropy (uncertainty sampling)
- DiversityAcquisition (k-means++ on embeddings)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random
from scipy.spatial.distance import cdist
from scipy.special import erf

from apedp.neural_discovery.architectures import init_model
from apedp.neural_discovery.config import AcquisitionType, NeuralDiscoveryConfig
from apedp.neural_discovery.training import NeuralDiscoveryTrainer
from apedp.neural_discovery.uncertainty import (
    ConformalPredictor,
    predict_with_uncertainty,
)

Array = jnp.ndarray
PRNGKey = jax.Array
Variables = dict[str, Array]


@dataclass
class AcquisitionResult:
    """Result of acquisition function."""
    indices: np.ndarray  # Selected indices
    scores: np.ndarray   # Acquisition scores
    metadata: dict[str, Any]


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    def __init__(self, config: NeuralDiscoveryConfig):
        self.config = config
        self.batch_size = config.acquisition_batch_size
        self.diversity_weight = config.acquisition_diversity_weight

    @abstractmethod
    def score(self,
              model: nn.Module,
              params: Variables,
              candidates: Array,
              **kwargs) -> np.ndarray:
        """Score candidate points for acquisition.

        Args:
            model: Trained model
            params: Model parameters
            candidates: Candidate inputs (n_candidates, n_features)

        Returns:
            Scores for each candidate (higher = more desirable)
        """

    def select(self,
               model: nn.Module,
               params: Variables,
               candidates: Array,
               n_select: int | None = None,
               **kwargs) -> AcquisitionResult:
        """Select top candidates."""
        n_select = n_select or self.batch_size
        scores = self.score(model, params, candidates, **kwargs)

        # Apply diversity if weight > 0
        if self.diversity_weight > 0:
            scores = self._apply_diversity(candidates, scores)

        # Select top-k
        indices = np.argsort(scores)[-n_select:][::-1]
        return AcquisitionResult(
            indices=indices,
            scores=scores[indices],
            metadata={'n_candidates': len(candidates), 'n_selected': n_select}
        )

    def _apply_diversity(self, candidates: Array, scores: np.ndarray) -> np.ndarray:
        """Apply diversity penalty using k-means++ style selection."""
        n = len(candidates)
        if n <= self.batch_size:
            return scores

        # Convert candidates to numpy for indexing
        candidates_np = np.array(candidates)

        # Normalize scores to [0, 1]
        score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Diversity: distance to already selected points
        selected = []
        remaining = list(range(n))

        for _ in range(min(self.batch_size, n)):
            if not remaining:
                break

            # Compute diversity scores
            if selected:
                selected_pts = candidates_np[selected]
                remaining_pts = candidates_np[remaining]
                dists = cdist(remaining_pts, selected_pts)
                min_dists = np.min(dists, axis=1)
                # Normalize
                div_norm = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min() + 1e-8)
                combined = (1 - self.diversity_weight) * score_norm[remaining] + self.diversity_weight * div_norm
                best_idx = remaining[np.argmax(combined)]
            else:
                best_idx = remaining[np.argmax(score_norm[remaining])]

            selected.append(best_idx)
            remaining.remove(best_idx)

        # Return scores only for selected
        result_scores = np.full(n, -np.inf)
        result_scores[selected] = scores[selected]
        return result_scores


class BALD(AcquisitionFunction):
    """Bayesian Active Learning by Disagreement.

    Measures mutual information between predictions and model parameters.
    High BALD = model is uncertain AND disagrees across ensemble members.

    For regression, uses variance-based BALD: Var[E[y|x,w]] (epistemic variance).
    """

    def score(self,
              model: nn.Module,
              params: Variables,
              candidates: Array,
              ensemble_params: list[Variables] | None = None,
              n_mc_samples: int = 30,
              **kwargs) -> np.ndarray:
        """BALD score for regression: epistemic variance.

        For regression, BALD ≈ Var_w[E[y|x,w]] = variance of ensemble means.
        """
        if ensemble_params is not None:
            # Ensemble-based BALD: variance of predictions across ensemble
            preds = []
            for ep in ensemble_params:
                pred = model.apply({'params': ep}, candidates)
                preds.append(pred)
            preds = jnp.stack(preds, axis=0)  # (ensemble, batch, output)

            # Epistemic variance = variance across ensemble members
            epistemic_var = jnp.var(preds, axis=0)  # (batch, output)
            # Mean over output dimension if multi-output
            if epistemic_var.ndim > 1:
                epistemic_var = jnp.mean(epistemic_var, axis=-1)
            return np.array(epistemic_var)

        else:
            output = predict_with_uncertainty(
                model, params, candidates, method="dropout",
                n_samples=n_mc_samples,
            )
            return np.asarray(output.std).reshape(len(candidates), -1).mean(axis=1)


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement for optimization.

    Standard acquisition function for Bayesian optimization.
    """

    def __init__(self, config: NeuralDiscoveryConfig, xi: float = 0.01):
        super().__init__(config)
        self.xi = xi  # Exploration parameter

    def score(self,
              model: nn.Module,
              params: Variables,
              candidates: Array,
              y_best: float,
              ensemble_params: list[Variables] | None = None,
              conformal: ConformalPredictor | None = None,
              **kwargs) -> np.ndarray:
        """Expected Improvement: E[max(0, y_best - y(x))].

        For minimization (lower is better).
        """
        # Get predictions with uncertainty
        if ensemble_params:
            output = predict_with_uncertainty(model, params, candidates,
                                              method="ensemble",
                                              ensemble_params=ensemble_params)
        elif conformal:
            output = conformal.predict(model, params, candidates)
        else:
            # Single model with MC dropout
            output = predict_with_uncertainty(model, params, candidates,
                                              method="dropout",
                                              n_samples=self.config.dropout_mc_samples)

        mean = np.array(output.mean).ravel()
        std = np.array(output.std).ravel()

        # EI for minimization
        z = (y_best - mean - self.xi) / (std + 1e-8)
        ei = (y_best - mean - self.xi) * norm_cdf(z) + std * norm_pdf(z)
        ei = np.maximum(ei, 0)

        return ei


class MaxVariance(AcquisitionFunction):
    """Max Variance: Select points with highest predictive variance."""

    def score(self,
              model: nn.Module,
              params: Variables,
              candidates: Array,
              ensemble_params: list[Variables] | None = None,
              **kwargs) -> np.ndarray:
        """Score = predictive variance (epistemic uncertainty)."""
        if ensemble_params:
            output = predict_with_uncertainty(model, params, candidates,
                                              method="ensemble",
                                              ensemble_params=ensemble_params)
        else:
            output = predict_with_uncertainty(model, params, candidates,
                                              method="dropout",
                                              n_samples=self.config.dropout_mc_samples)

        return np.array(output.std).ravel()


class MaxEntropy(AcquisitionFunction):
    """Max Entropy: Select points with highest predictive entropy."""

    def score(self,
              model: nn.Module,
              params: Variables,
              candidates: Array,
              ensemble_params: list[Variables] | None = None,
              **kwargs) -> np.ndarray:
        """Score = predictive entropy (Gaussian approximation)."""
        if ensemble_params:
            output = predict_with_uncertainty(model, params, candidates,
                                              method="ensemble",
                                              ensemble_params=ensemble_params)
        else:
            output = predict_with_uncertainty(model, params, candidates,
                                              method="dropout",
                                              n_samples=self.config.dropout_mc_samples)

        # Entropy of predictive distribution (Gaussian approximation)
        # H = 0.5 * log(2 * pi * e * var)
        var = np.array(output.std) ** 2
        entropy = 0.5 * np.log(2 * np.pi * np.e * var + 1e-8)
        return np.array(entropy).ravel()


class DiversityAcquisition(AcquisitionFunction):
    """Diversity-based acquisition using k-means++ on embeddings.

    Selects points that cover the input space well.
    """

    def __init__(self, config: NeuralDiscoveryConfig,
                 embedding_model: nn.Module | None = None,
                 embedding_params: Variables | None = None):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.embedding_params = embedding_params

    def score(self,
              model: nn.Module,
              params: Variables,
              candidates: Array,
              already_selected: np.ndarray | None = None,
              **kwargs) -> np.ndarray:
        """Score = minimum distance to already selected points."""
        # Get embeddings
        if self.embedding_model and self.embedding_params:
            embeddings = self.embedding_model.apply(
                {'params': self.embedding_params}, candidates)
        else:
            # Use input features directly
            embeddings = candidates

        embeddings = np.array(embeddings)

        if already_selected is not None and len(already_selected) > 0:
            selected_emb = embeddings[already_selected]
            dists = cdist(embeddings, selected_emb)
            min_dists = np.min(dists, axis=1)
            return min_dists
        else:
            # First point: farthest from center
            center = np.mean(embeddings, axis=0)
            dists = np.linalg.norm(embeddings - center, axis=1)
            return dists


class ActiveLearningLoop:
    """Orchestrates active learning: train -> acquire -> label -> retrain."""

    def __init__(self,
                 model_fn: Callable[[], nn.Module],
                 config: NeuralDiscoveryConfig,
                 acquisition_fn: AcquisitionFunction):
        self.model_fn = model_fn
        self.config = config
        self.acquisition_fn = acquisition_fn
        self.history = []

    def run(self,
            initial_labeled: tuple[Array, Array],
            unlabeled_pool: Array,
            labeling_fn: Callable[[Array], Array],
            n_rounds: int,
            key: PRNGKey) -> dict[str, Any]:
        """Run active learning loop.

        Args:
            initial_labeled: (x_labeled, y_labeled)
            unlabeled_pool: Unlabeled candidate points
            labeling_fn: Function to get labels for selected points (simulates lab)
            n_rounds: Number of active learning rounds
            key: PRNG key

        Returns:
            History of selections and performance
        """
        x_labeled, y_labeled = initial_labeled
        pool = unlabeled_pool.copy()

        for round_idx in range(n_rounds):
            print(f"Active Learning Round {round_idx + 1}/{n_rounds}")

            # Train model on current labeled data
            model = self.model_fn()
            train_key = random.fold_in(key, round_idx)
            trained_model, trained_params = self._train_model(model, x_labeled, y_labeled, train_key)

            # Select next points
            acquisition_result = self.acquisition_fn.select(
                trained_model, trained_params, pool
            )
            selected_indices = acquisition_result.indices
            selected_points = pool[selected_indices]

            # Get labels (simulate experiment)
            selected_labels = labeling_fn(selected_points)

            # Update labeled set
            x_labeled = np.vstack([x_labeled, selected_points])
            y_labeled = np.concatenate([y_labeled, selected_labels])

            # Remove from pool
            pool = np.delete(pool, selected_indices, axis=0)

            # Record history
            self.history.append({
                'round': round_idx,
                'selected_indices': selected_indices.tolist(),
                'selected_points': selected_points.tolist(),
                'acquisition_scores': acquisition_result.scores.tolist(),
                'n_labeled': len(x_labeled),
                'n_pool': len(pool),
            })

            print(f"  Selected {len(selected_indices)} points, pool size: {len(pool)}")

            if len(pool) == 0:
                break

        return {
            'final_labeled': (x_labeled, y_labeled),
            'final_pool': pool,
            'history': self.history,
        }

    def _train_model(self, model: nn.Module, x: Array, y: Array,
                     key: PRNGKey) -> tuple[nn.Module, Variables]:
        """Train a fresh model on the currently labeled data."""
        params = init_model(model, x.shape, key, self.config)
        trainer = NeuralDiscoveryTrainer(self.config)
        params, _ = trainer.train(
            model,
            (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)),
            val_ds=None,
            key=key,
            verbose=False,
        )
        return model, params


def create_acquisition_function(config: NeuralDiscoveryConfig,
                                acquisition_type: AcquisitionType | None = None) -> AcquisitionFunction:
    """Factory for acquisition functions."""
    acq_type = acquisition_type or config.acquisition_type

    if acq_type == "bald":
        return BALD(config)
    elif acq_type == "expected_improvement":
        return ExpectedImprovement(config)
    elif acq_type == "max_variance":
        return MaxVariance(config)
    elif acq_type == "max_entropy":
        return MaxEntropy(config)
    elif acq_type == "diversity":
        return DiversityAcquisition(config)
    else:
        raise ValueError(f"Unknown acquisition type: {acq_type}")


# Helper functions for EI
def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


__all__ = [
    "BALD",
    "AcquisitionFunction",
    "AcquisitionResult",
    "ActiveLearningLoop",
    "DiversityAcquisition",
    "ExpectedImprovement",
    "MaxEntropy",
    "MaxVariance",
    "create_acquisition_function",
]