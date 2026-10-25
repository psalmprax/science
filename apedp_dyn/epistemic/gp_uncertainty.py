"""Gaussian Process uncertainty quantification for discovered equations.

Provides GP surrogates for Bayesian optimization guidance and uncertainty
estimates to distinguish interpolative fits from extrapolative predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class GPPrediction:
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray


class RBFKernel:
    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0):
        self.length_scale = length_scale
        self.sigma_f = sigma_f

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        # compute squared distance
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.length_scale**2 * sqdist)


class GaussianProcessRegressor:
    def __init__(self, kernel: RBFKernel | None = None, sigma_n: float = 1e-6):
        self.kernel = kernel or RBFKernel()
        self.sigma_n = sigma_n
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X) + self.sigma_n**2 * np.eye(len(X))
        # Cholesky
        self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> GPPrediction:
        X = np.atleast_2d(X)
        K_s = self.kernel(self.X_train, X)
        mu_s = K_s.T @ self.alpha
        if return_std:
            v = np.linalg.solve(self.L, K_s)
            K_ss = self.kernel(X, X)
            var_s = np.diag(K_ss) - np.sum(v**2, axis=0)
            var_s = np.maximum(var_s, 1e-12)
            std_s = np.sqrt(var_s)
        else:
            std_s = np.zeros_like(mu_s)
            var_s = np.zeros_like(mu_s)
        return GPPrediction(mean=mu_s, std=std_s, variance=var_s)

    def log_marginal_likelihood(self) -> float:
        if self.L is None:
            return -np.inf
        y = self.y_train
        return -0.5 * y @ self.alpha - np.sum(np.log(np.diag(self.L))) - 0.5 * len(y) * np.log(2 * np.pi)


class BayesianOptimizer:
    def __init__(self, gp: GaussianProcessRegressor, X_bounds: np.ndarray):
        self.gp = gp
        self.bounds = np.atleast_2d(X_bounds)

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        pred = self.gp.predict(X)
        mu = pred.mean
        sigma = pred.std
        mu_sample_opt = np.max(self.gp.y_train)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            from scipy.stats import norm
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
