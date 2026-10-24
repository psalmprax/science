"""Riemann Zeta Zeros Dataset & Feature Transformation Engine.

Generates and caches high-precision non-trivial zeros of the Riemann zeta function
rho_n = beta_n + i * gamma_n, providing coordinate representations, derived invariants,
and strict out-of-distribution (OOD) test splits.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mpmath
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "zeta_zeros_500.json"


def generate_or_load_zeta_zeros(n_zeros: int = 200) -> list[dict[str, float]]:
    """Loads pre-cached zeta zeros or computes them via mpmath.zetazero."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if len(data) >= n_zeros:
                return data[:n_zeros]  # type: ignore[no-any-return]
        except Exception:
            logger.debug("cached zeta zeros load failed, recomputing", exc_info=True)

    # Compute high-precision zeros
    zeros: list[dict[str, float]] = []
    mpmath.mp.dps = 25
    prev_gamma = 0.0
    for idx in range(1, n_zeros + 1):
        z = mpmath.zetazero(idx)
        beta = float(z.real)
        gamma = float(z.imag)
        spacing = gamma - prev_gamma if prev_gamma > 0 else 0.0
        prev_gamma = gamma
        norm = float(abs(z))
        arg = float(mpmath.arg(z))
        zeros.append({
            "n": float(idx),
            "beta": beta,
            "gamma": gamma,
            "spacing": spacing,
            "norm": norm,
            "arg": arg,
        })

    # Cache computed zeros
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(zeros, f, indent=2)
    except Exception:
        logger.debug("caching zeta zeros failed", exc_info=True)

    return zeros


class RiemannZetaBenchmarkDataset:
    """Constructs train, validation, and OOD splits for RH Discovery Variants."""

    def __init__(self, n_total_zeros: int = 200, train_n: int = 100, val_n: int = 50):
        self.zeros = generate_or_load_zeta_zeros(n_total_zeros)
        self.train_n = train_n
        self.val_n = val_n
        self.ood_n = len(self.zeros) - (train_n + val_n)

        # Convert to numpy arrays
        self.n_arr = np.array([z["n"] for z in self.zeros], dtype=float)
        self.beta_arr = np.array([z["beta"] for z in self.zeros], dtype=float)
        self.gamma_arr = np.array([z["gamma"] for z in self.zeros], dtype=float)
        self.spacing_arr = np.array([z["spacing"] for z in self.zeros], dtype=float)
        self.norm_arr = np.array([z["norm"] for z in self.zeros], dtype=float)
        self.arg_arr = np.array([z["arg"] for z in self.zeros], dtype=float)

    def get_variant_a_full_zeros(self) -> dict[str, dict[str, np.ndarray]]:
        """Variant A: Full coordinates (n, gamma_n) -> target beta_n."""
        cut1 = self.train_n
        cut2 = self.train_n + self.val_n

        return {
            "train": {"n": self.n_arr[:cut1], "gamma": self.gamma_arr[:cut1], "beta": self.beta_arr[:cut1]},
            "val": {"n": self.n_arr[cut1:cut2], "gamma": self.gamma_arr[cut1:cut2], "beta": self.beta_arr[cut1:cut2]},
            "ood": {"n": self.n_arr[cut2:], "gamma": self.gamma_arr[cut2:], "beta": self.beta_arr[cut2:]},
        }

    def get_variant_b_transformed(self) -> dict[str, dict[str, np.ndarray]]:
        """Variant B: Transformed coordinates u_n = 2*beta_n - 1 -> target u_n = 0.0."""
        u_arr = 2.0 * self.beta_arr - 1.0
        cut1 = self.train_n
        cut2 = self.train_n + self.val_n

        return {
            "train": {"gamma": self.gamma_arr[:cut1], "u": u_arr[:cut1]},
            "val": {"gamma": self.gamma_arr[cut1:cut2], "u": u_arr[cut1:cut2]},
            "ood": {"gamma": self.gamma_arr[cut2:], "u": u_arr[cut2:]},
        }

    def get_variant_c_invariants(self) -> dict[str, dict[str, np.ndarray]]:
        """Variant C: Derived Invariants norm, gamma, spacing -> target beta_n."""
        cut1 = self.train_n
        cut2 = self.train_n + self.val_n

        return {
            "train": {
                "norm": self.norm_arr[:cut1],
                "gamma": self.gamma_arr[:cut1],
                "spacing": self.spacing_arr[:cut1],
                "beta": self.beta_arr[:cut1],
            },
            "val": {
                "norm": self.norm_arr[cut1:cut2],
                "gamma": self.gamma_arr[cut1:cut2],
                "spacing": self.spacing_arr[cut1:cut2],
                "beta": self.beta_arr[cut1:cut2],
            },
            "ood": {
                "norm": self.norm_arr[cut2:],
                "gamma": self.gamma_arr[cut2:],
                "spacing": self.spacing_arr[cut2:],
                "beta": self.beta_arr[cut2:],
            },
        }
