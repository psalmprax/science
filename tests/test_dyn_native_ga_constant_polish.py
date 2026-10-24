"""Native GA constant polish pins (coordinate-descent golden-section stage).

The polish stage (`genetic.rs::polish_candidate_constants`) refines a
candidate's inner `Const` leaves after evolution, refitting the affine
readout at every probe and replacing a constant only on a strict full-pipeline
MSE gain. Mechanism-level behavior (sin(c*x): c = 1.8 -> 2.000, >100x MSE
collapse; joint (frequency, phase) convergence; constant-free trees untouched)
is pinned in Rust unit tests.

Scope note (measured 2026-10-16): the polish never worsens the best candidate,
but end-to-end gains on hard math targets are bounded by structure discovery
-- the native GA at default budget does not find the sin/exp families on
those targets, so there is nothing for the polish to refine. This file pins
the plumbing (flag forwarding, never-worsens, eval accounting), not a
structure-recovery claim.
"""
from __future__ import annotations

import numpy as np

import apedp_kernel as ak
from apedp.rust_bridge import RustKernelBridge


def _best(raw) -> float:
    return min(item[1] for item in raw)


def test_polish_never_worsens_best_and_counts_evals():
    x = np.linspace(0.05, 3.0, 60)
    y = np.sin(2.0 * x)
    args = (
        [np.ascontiguousarray(x)],
        ["x"],
        np.ascontiguousarray(y),
        60,
        12,
        2,
        5,
        4,
        4,
        0.35,
        0.70,
        True,
        True,
        True,
        True,
    )
    raw_off, evals_off = ak.evolve_native(*args, False, 3, 42)
    raw_on, evals_on = ak.evolve_native(*args, True, 3, 42)
    assert raw_off and raw_on
    assert _best(raw_on) <= _best(raw_off) + 1e-12
    assert evals_on >= evals_off


def test_polish_top_k_zero_equals_disabled():
    x = np.linspace(0.05, 3.0, 60)
    y = np.sin(2.0 * x)
    args = (
        [np.ascontiguousarray(x)],
        ["x"],
        np.ascontiguousarray(y),
        40,
        8,
        2,
        5,
        4,
        4,
        0.35,
        0.70,
        True,
        True,
        True,
        True,
    )
    raw_off, evals_off = ak.evolve_native(*args, False, 3, 42)
    raw_zero, evals_zero = ak.evolve_native(*args, True, 0, 42)
    assert _best(raw_zero) == _best(raw_off)
    assert evals_zero == evals_off


def test_bridge_forwards_polish_flags():
    bridge = RustKernelBridge()
    x = np.linspace(-3, 3, 200)
    y = 3.0 * x ** 2 + 2.0
    common = {
        "population_size": 50,
        "generations": 20,
        "islands": 4,
        "seed": 42,
    }
    pareto_off, evals_off = bridge.evolve_native(
        [x], ["x"], y, polish_constants=False, **common
    )
    pareto_on, evals_on = bridge.evolve_native(
        [x], ["x"], y, polish_constants=True, polish_top_k=3, **common
    )
    best_off = min(p["mse"] for p in pareto_off)
    best_on = min(p["mse"] for p in pareto_on)
    assert best_on <= best_off + 1e-12
    assert best_on < 1e-6, f"x^2 recovery must survive polish: {best_on}"
    assert evals_on >= evals_off
