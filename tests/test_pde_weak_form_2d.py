"""The 2-D weak form: integrals by parts are right, known equations come back, and small real terms are kept."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pde_2d_worlds as worlds

from apedp_dyn.pde import weak_form_2d as wf


def test_weak_system_shapes_and_the_exact_solution_satisfies_it():
    """u = exp(-2 D t) sin(x) sin(y) solves the heat equation: the weak rows must balance with the true coefficients."""
    n, frames, D = 48, 61, 0.1
    x = np.arange(n) * 2 * np.pi / n
    t = np.linspace(0.0, 1.0, frames)
    u = np.exp(-2 * D * t)[:, None, None] * np.sin(x)[None, :, None] * np.sin(x)[None, None, :]
    G, b = wf.weak_system(u, t[1] - t[0], x[1] - x[0], x[1] - x[0])
    names = [name for name, *_ in wf.LIBRARY_2D]
    assert G.shape == (len(b), len(names)) and len(b) > 100
    w = np.zeros(len(names))
    w[names.index("u_xx")] = w[names.index("u_yy")] = D
    assert np.linalg.norm(G @ w - b) / np.linalg.norm(b) < 1e-3


@pytest.mark.parametrize("world", ["heat", "advection_diffusion", "burgers"])
def test_known_equations_are_recovered_with_five_percent_noise(world):
    sim = worlds.simulate(world, seed=900, frames=61)
    found = wf.identify(worlds.with_noise(sim["u"], 0.05, 901), sim["dt"], sim["dx"], sim["dy"])
    assert set(found) == set(sim["truth"])
    assert max(abs(found[k] - v) / abs(v) for k, v in sim["truth"].items()) < 0.05
    quality = wf.weak_fit_quality(sim["u"], sim["dt"], sim["dx"], sim["dy"], found)
    assert quality["r2"] > 0.99


def test_evidence_rule_keeps_a_small_real_term_the_threshold_rule_drops():
    sim = worlds.simulate("fisher_kpp", seed=900)
    by_evidence = wf.identify(sim["u"], sim["dt"], sim["dx"], sim["dy"])
    by_threshold = wf.identify(sim["u"], sim["dt"], sim["dx"], sim["dy"], select="threshold")
    assert set(by_evidence) == set(sim["truth"])
    assert set(by_threshold) < set(sim["truth"])  # reaction dominates, so a diffusion term falls under 5 %


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        wf.weak_system(np.zeros((10, 10)), 0.1, 0.1, 0.1)
    with pytest.raises(ValueError):
        wf.identify(np.zeros((30, 20, 20)), 0.1, 0.1, 0.1, select="guess")
