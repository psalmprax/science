"""End-to-end pins for GA trig frequency/phase template seeding.

Frozen N04 forensics (docs/frontier_9_of_10_architectural_resolution.md):
uniform random trees at depth 4 could not express `sin(f*x + p)`, so every
oscillatory target collapsed to mush (family accuracy 0 on this class).
`GeneticOps::random_tree` now seeds the template at top level (rate 0.18)
and the polish stage escape-couples (f, p) via a Powell-style conjugate
direction. Capability pinned here; seeding distribution and flag gating are
pinned in `genetic.rs::trig_seed_tests`.
"""
from __future__ import annotations

import numpy as np

import apedp_kernel as ak


def _run(y: np.ndarray) -> tuple[float, str]:
    x = np.linspace(0.05, 3.0, 60)
    raw, _ = ak.evolve_native(
        [np.ascontiguousarray(x)], ["x"], np.ascontiguousarray(y),
        100, 25, 2, 5, 4, 4, 0.35, 0.70, True, True, True, True, True, 3, 42,
    )
    best = min(raw, key=lambda m: m[1])
    return best[1], best[0]


def test_sin_family_exact_recovery():
    x = np.linspace(0.05, 3.0, 60)
    mse, expr = _run(np.sin(2.0 * x))
    assert mse < 1e-12, (mse, expr)


def test_cos_family_exact_recovery():
    x = np.linspace(0.05, 3.0, 60)
    mse, expr = _run(2.0 * np.cos(1.3 * x))
    assert mse < 1e-12, (mse, expr)


def test_phase_shifted_trig_reaches_coupled_constants():
    x = np.linspace(0.05, 3.0, 60)
    mse, expr = _run(3.0 * np.sin(1.7 * x + 0.4))
    # Family correct with frequency/phase resolved to ~5e-3;
    # full collapse needs the Powell conjugate direction in the polish stage.
    assert mse < 1e-5, (mse, expr)
    assert "cos" in expr or "sin" in expr, expr
