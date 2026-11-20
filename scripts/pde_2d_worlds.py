"""Simulated 2-D fields with known equations, for testing the 2-D weak form. Periodic, pseudo-spectral, RK4.

Each world is ``u_t = sum of terms`` written in the library of ``apedp_dyn.pde.weak_form_2d``; ``truth``
gives the coefficients. ``simulate`` returns the field sampled on a uniform grid with its spacings.
"""
from __future__ import annotations

from typing import Any

import numpy as np

WORLDS: dict[str, dict[str, float]] = {
    "heat": {"u_xx": 0.10, "u_yy": 0.10},
    "advection_diffusion": {"u_x": -0.8, "u_y": 0.5, "u_xx": 0.05, "u_yy": 0.05},
    "fisher_kpp": {"u": 1.0, "u^2": -1.0, "u_xx": 0.05, "u_yy": 0.05},
    "allen_cahn": {"u": 1.0, "u^3": -1.0, "u_xx": 0.03, "u_yy": 0.03},
    "burgers": {"(u^2)_x": -0.5, "(u^2)_y": -0.5, "u_xx": 0.08, "u_yy": 0.08},
    "anisotropic_diffusion": {"u_xx": 0.12, "u_yy": 0.03},
}
_TERMS = {"u": (1, 0, 0), "u^2": (2, 0, 0), "u^3": (3, 0, 0), "u_x": (1, 1, 0), "u_y": (1, 0, 1), "(u^2)_x": (2, 1, 0),
          "(u^2)_y": (2, 0, 1), "u_xx": (1, 2, 0), "u_yy": (1, 0, 2), "(u^2)_xx": (2, 2, 0), "(u^2)_yy": (2, 0, 2)}


def initial(world: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth random field from the lowest few Fourier modes, scaled to suit the world."""
    x = np.arange(n) * 2 * np.pi / n
    mesh: list[np.ndarray] = list(np.meshgrid(x, x, indexing="ij"))
    X, Y = mesh
    u: np.ndarray = np.zeros((n, n))
    for kx in range(-3, 4):
        for ky in range(-3, 4):
            if kx == ky == 0:
                continue
            amplitude = rng.standard_normal() / (1.0 + kx * kx + ky * ky)
            u += amplitude * np.cos(kx * X + ky * Y + rng.uniform(0, 2 * np.pi))
    u /= np.abs(u).max()
    if world == "fisher_kpp":
        u = 0.5 + 0.4 * u  # a population between 0 and 1
    elif world == "allen_cahn":
        u = 0.9 * u
    elif world == "burgers":
        u = u + 0.5
    return u


def simulate(world: str, seed: int, *, n: int = 64, frames: int = 101, t_end: float = 1.0, substeps: int = 20) -> dict[str, Any]:
    truth = WORLDS[world]
    rng = np.random.default_rng(seed)
    u = initial(world, n, rng)
    k = np.fft.fftfreq(n, d=1.0 / n)
    grids: list[np.ndarray] = list(np.meshgrid(k, k, indexing="ij"))
    KX, KY = grids
    mask = (np.abs(KX) < n / 3) & (np.abs(KY) < n / 3)  # 2/3 dealiasing for the products

    def rhs(v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        for name, w in truth.items():
            p, a, b = _TERMS[name]
            spectrum = np.fft.fft2(v ** p) * (mask if p > 1 else 1.0)
            out += w * np.real(np.fft.ifft2((1j * KX) ** a * (1j * KY) ** b * spectrum))
        return out

    dt_frame = t_end / (frames - 1)
    h = dt_frame / substeps
    field = np.empty((frames, n, n))
    field[0] = u
    for f in range(1, frames):
        for _ in range(substeps):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * h * k1)
            k3 = rhs(u + 0.5 * h * k2)
            k4 = rhs(u + h * k3)
            u = u + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        field[f] = u
    if not np.all(np.isfinite(field)):
        raise FloatingPointError(f"{world} seed {seed} did not stay finite")
    return {"u": field, "dt": dt_frame, "dx": 2 * np.pi / n, "dy": 2 * np.pi / n, "truth": dict(truth)}


def with_noise(u: np.ndarray, level: float, seed: int) -> np.ndarray:
    """Additive Gaussian noise with standard deviation ``level`` times the field's."""
    if level <= 0:
        return u
    noisy: np.ndarray = u + level * float(np.std(u)) * np.random.default_rng(seed).standard_normal(u.shape)
    return noisy
