"""Bayesian active learning for guiding discovery search.

Uses GP uncertainty + acquisition functions to prioritize experiments/candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gp_uncertainty import GaussianProcessRegressor, BayesianOptimizer


@dataclass
class ActiveLearningSuggestion:
    x_next: np.ndarray
    acquisition_value: float
    uncertainty: float
    reason: str


class BayesianActiveLearner:
    def __init__(self, gp: GaussianProcessRegressor | None = None, bounds: np.ndarray | None = None):
        self.gp = gp or GaussianProcessRegressor()
        self.bounds = bounds
        self.X_observed = []
        self.y_observed = []

    def update(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        if self.X_observed:
            self.X_observed = np.vstack([self.X_observed, X])
            self.y_observed = np.concatenate([self.y_observed, y])
        else:
            self.X_observed = X
            self.y_observed = y
        self.gp.fit(self.X_observed, self.y_observed)
        return self

    def suggest_next(self, n_points: int = 1, acquisition: str = "ei") -> list[ActiveLearningSuggestion]:
        suggestions = []
        if self.bounds is None or len(self.X_observed) == 0:
            return suggestions
        d = self.bounds.shape[0]
        # random sampling + optimization
        rng = np.random.RandomState(123)
        candidates = rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1000, d))
        if acquisition == "ei":
            bo = BayesianOptimizer(self.gp, self.bounds)
            acq = bo.expected_improvement(candidates)
        else:  # uncertainty sampling
            pred = self.gp.predict(candidates)
            acq = pred.std
        # pick top
        idx = np.argsort(acq)[-n_points:][::-1]
        for i in idx:
            x = candidates[i]
            pred = self.gp.predict(x.reshape(1, -1))
            suggestions.append(ActiveLearningSuggestion(
                x_next=x,
                acquisition_value=float(acq[i]),
                uncertainty=float(pred.std[0]),
                reason=f"{acquisition}_acquisition",
            ))
        return suggestions

    def prioritize_candidates(self, candidate_features: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
        """Prioritize candidates using uncertainty if scores missing."""
        candidate_features = np.atleast_2d(candidate_features)
        if self.X_observed is not None and len(self.X_observed) > 0:
            pred = self.gp.predict(candidate_features)
            # high uncertainty first for exploration
            priority = pred.std
            return np.argsort(priority)[::-1]
        if scores is not None:
            return np.argsort(scores)[::-1]
        return np.arange(len(candidate_features))
