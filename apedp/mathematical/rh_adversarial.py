"""Adversarial Worlds & Fake-RH Generator for Riemann Benchmark.

Constructs synthetic adversarial alternative worlds, fake-RH counter-benchmarks,
and precision/noise variants to rigorously test identifiability, ambiguity detection,
and anti-gaming / anti-overfitting safeguards.
"""
from __future__ import annotations

import numpy as np


class RiemannAdversarialEngine:
    """Generates synthetic adversarial datasets and fake critical-line worlds."""

    def __init__(self, base_n: np.ndarray, base_gamma: np.ndarray):
        self.n_arr = np.asarray(base_n, dtype=float)
        self.gamma_arr = np.asarray(base_gamma, dtype=float)
        self.num_points = len(self.n_arr)

    # --------------------------------------------------------------------------
    # Stage RH-2: Five Adversarial Alternatives
    # --------------------------------------------------------------------------
    def get_alternative_a(self) -> dict[str, np.ndarray]:
        """Alternative A: beta_n = 0.5 + 1e-3 (distinguishable offset)."""
        beta = 0.5 + 1e-3 * np.ones_like(self.n_arr)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_alternative_b(self) -> dict[str, np.ndarray]:
        """Alternative B: beta_n = 0.5 + 1e-6 (small but distinguishable offset)."""
        beta = 0.5 + 1e-6 * np.ones_like(self.n_arr)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_alternative_c(self) -> dict[str, np.ndarray]:
        """Alternative C: beta_n = 0.5 + 1 / n^2 (asymptotic convergence to 1/2)."""
        beta = 0.5 + 1.0 / (self.n_arr ** 2)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_alternative_d(self) -> dict[str, np.ndarray]:
        """Alternative D: beta_n = 0.5 + 1e-12 * sin(n) (at numerical precision boundary)."""
        beta = 0.5 + 1e-12 * np.sin(self.n_arr)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_alternative_e(self, split_n: int = 80, shift: float = 0.05) -> dict[str, np.ndarray]:
        """Alternative E: beta_n = 0.5 for n < split_n, 0.5 + shift for n >= split_n."""
        beta = np.where(self.n_arr < split_n, 0.5, 0.5 + shift)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    # --------------------------------------------------------------------------
    # Fake-RH Counter-Benchmarks
    # --------------------------------------------------------------------------
    def get_fake_rh_1(self) -> dict[str, np.ndarray]:
        """Fake-RH-1: beta_n = 0.5 + 1e-3 * sin(n) (oscillatory perturbation)."""
        beta = 0.5 + 1e-3 * np.sin(self.n_arr)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_fake_rh_2(self) -> dict[str, np.ndarray]:
        """Fake-RH-2: beta_n = 0.5 + 1 / n (slow harmonic decay)."""
        beta = 0.5 + 1.0 / self.n_arr
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_fake_rh_3(self, split_n: int = 80, ood_val: float = 0.51) -> dict[str, np.ndarray]:
        """Fake-RH-3: beta_n = 0.5 (n <= split_n) but beta_n = 0.51 (n > split_n)."""
        beta = np.where(self.n_arr <= split_n, 0.5, ood_val)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    # --------------------------------------------------------------------------
    # Expanded Adversaries (Symbolic Regression Stress Testing)
    # --------------------------------------------------------------------------
    def get_adversary_local_agreement(self, epsilon: float = 1e-4, n_obs_max: int = 100) -> dict[str, np.ndarray]:
        """Local Agreement: beta_n = 0.5 + eps * (n / n_obs_max)^4 (diverges far outside observed range)."""
        beta = 0.5 + epsilon * ((self.n_arr / float(n_obs_max)) ** 4)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_adversary_delayed_divergence(self, split_n: int = 100, shift: float = 0.05) -> dict[str, np.ndarray]:
        """Delayed Divergence: beta_n = 0.5 for n < split_n, 0.5 + shift for n >= split_n."""
        beta = np.where(self.n_arr < split_n, 0.5, 0.5 + shift)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_adversary_oscillatory_cancellation(self, epsilon: float = 1e-3) -> dict[str, np.ndarray]:
        """Oscillatory Cancellation: beta_n = 0.5 + eps * sin(log(n + 1))."""
        beta = 0.5 + epsilon * np.sin(np.log(self.n_arr + 1.0))
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_adversary_slowly_vanishing_deviation(self) -> dict[str, np.ndarray]:
        """Slowly Vanishing Deviation: beta_n = 0.5 + 1 / log(n + 2)."""
        beta = 0.5 + 1.0 / np.log(self.n_arr + 2.0)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    def get_adversary_tiny_deviation(self, k: int = 8) -> dict[str, np.ndarray]:
        """Adversarially Tiny Deviation: beta_n = 0.5 + 10^(-k) * cos(n)."""
        beta = 0.5 + (10.0 ** (-k)) * np.cos(self.n_arr)
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}

    # --------------------------------------------------------------------------
    # Precision / Noise Variants
    # --------------------------------------------------------------------------
    def get_precision_variant(self, noise_scale: float, seed: int = 42) -> dict[str, np.ndarray]:
        """Generates true critical-line zeros with specified observation noise scale."""
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_scale, size=self.num_points) if noise_scale > 0 else np.zeros(self.num_points)
        beta = 0.5 * np.ones(self.num_points) + noise
        return {"n": self.n_arr, "gamma": self.gamma_arr, "beta": beta}
