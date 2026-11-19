"""Development study: weak-form PDE identification under noise, beside the pointwise kernel.

    python scripts/study_weak_form_pde.py

NOT a preregistered result. Two 1-D fields: a heat equation u_t = 0.8 u_xx
(closed form, two Fourier modes) and a viscous Burgers equation
u_t = -0.5 (u^2)_x + 0.1 u_xx (pseudo-spectral, RK4). Noise is Gaussian at a
fraction of the field's standard deviation; 10 seeds per level. The weak form's
threshold (0.05), windows and test-function order are the module's defaults and
were not varied. The kernel (``discover_pde``) is run on the same fields for
the heat equation only: its library has no advection term, so Burgers is
outside what it can express and comparing them there would be unfair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.pde import weak_form  # noqa: E402
from apedp_dyn.pde.dispatch import discover_pde  # noqa: E402

OUT = ROOT / "benchmarks" / "weak_form_pde_dev_study.json"
NOISE = (0.0, 0.01, 0.05, 0.10, 0.20)
SEEDS = tuple(range(10))


def heat() -> tuple[np.ndarray, float, float]:
    nx, nt, dx, dt, diffusion = 64, 400, 0.1, 0.002, 0.8
    x, t = np.arange(nx) * dx, np.arange(nt) * dt
    k1, k2 = 2 * np.pi / (nx * dx), 4 * np.pi / (nx * dx)
    u = (np.exp(-diffusion * k1 ** 2 * t)[:, None] * np.sin(k1 * x)[None]
         + 0.5 * np.exp(-diffusion * k2 ** 2 * t)[:, None] * np.cos(k2 * x)[None])
    return u, dt, dx


def burgers() -> tuple[np.ndarray, float, float]:
    nx, nt, dt, nu, sub = 128, 400, 0.002, 0.1, 10
    dx = 2 * np.pi / nx
    x = np.arange(nx) * dx
    k = np.fft.fftfreq(nx, dx) * 2 * np.pi
    u = np.sin(x) + 0.5
    h = dt / sub

    def rhs(v: np.ndarray) -> np.ndarray:
        out: np.ndarray = (-0.5 * np.real(np.fft.ifft(1j * k * np.fft.fft(v * v)))
                           + nu * np.real(np.fft.ifft(-(k ** 2) * np.fft.fft(v))))
        return out

    frames = []
    for _ in range(nt):
        frames.append(u.copy())
        for _ in range(sub):
            a = rhs(u)
            b = rhs(u + h / 2 * a)
            c = rhs(u + h / 2 * b)
            d = rhs(u + h * c)
            u = u + h / 6 * (a + 2 * b + 2 * c + d)
    return np.array(frames), dt, dx


def main() -> int:
    cases = {"heat": (heat(), {"u_xx": 0.8}), "burgers": (burgers(), {"(u^2)_x": -0.5, "u_xx": 0.1})}
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, ((u, dt, dx), truth) in cases.items():
        summary[name] = {}
        for noise in NOISE:
            exact, errors, kernel_errors = 0, [], []
            for seed in SEEDS:
                noisy = u + np.random.default_rng(seed).normal(0.0, 1.0, u.shape) * noise * u.std()
                found = weak_form.identify(noisy, dt, dx)
                exact += set(found) == set(truth)
                errors.append(max(abs(found.get(k, 0.0) - v) / abs(v) for k, v in truth.items()))
                if name == "heat":
                    res = discover_pde(noisy, dt=dt, dxs=dx)
                    got = {t: float(c) for t, c in zip(res.active_terms, res.coefficients, strict=False)}
                    kernel_errors.append(abs(got.get("laplacian(u)", 0.0) - 0.8) / 0.8)
            cell: dict[str, float | int] = {"weak_exact_terms": exact, "of": len(SEEDS),
                                            "weak_median_worst_rel_coef_error": float(np.median(errors))}
            if name == "heat":
                cell["kernel_median_rel_error_of_diffusion"] = float(np.median(kernel_errors))
            summary[name][str(noise)] = cell
            print(name, noise, cell)
    OUT.write_text(json.dumps({"status": "development study, not preregistered", "summary": summary},
                              indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
