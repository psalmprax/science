"""Graph discovery: scoring, the baselines, the even-link pass, and (when Tigramite is present) PCMCI+ itself."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import causal_graph_worlds as worlds

from apedp_dyn.proposers import causal_graph as cg
from apedp_dyn.proposers.pcmci_proposer import is_available


def _chain(n=1500, seed=0):
    """x -> y (lag 1) -> z (lag 2); x also reaches z only through y."""
    rng = np.random.default_rng(seed)
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.8 * x[t - 1] + rng.standard_normal()
        z[t] = 0.8 * y[t - 2] + rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y, "z": z}), {("x", 1, "x"), ("x", 1, "y"), ("y", 2, "z")}


def test_score_counts_hits_precision_and_recall():
    out = cg.score({("a", 1, "b"), ("a", 2, "b")}, {("a", 1, "b"), ("c", 1, "b")})
    assert (out["hit"], out["precision"], out["recall"]) == (1.0, 0.5, 0.5)
    assert cg.score(set(), set())["f1"] == 1.0


def test_vector_autoregression_finds_the_direct_links_and_correlation_adds_indirect_ones():
    frame, truth = _chain()
    by_var = cg.as_set(cg.var_links(frame))
    by_correlation = cg.as_set(cg.correlation_links(frame))
    assert truth <= by_var and cg.score(by_var, truth)["precision"] >= 0.6
    assert ("x", 3, "z") in by_correlation and ("x", 3, "z") not in truth  # the indirect path, reported as a link
    assert cg.score(by_correlation, truth)["precision"] < cg.score(by_var, truth)["precision"]


def test_even_pass_finds_a_link_through_the_square_and_nothing_on_linear_data():
    rng = np.random.default_rng(1)
    n = 2000
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + rng.standard_normal()
        y[t] = 0.3 * y[t - 1] + 0.6 * (x[t - 1] ** 2 - 1.0) + rng.standard_normal()
    frame = pd.DataFrame({"x": x, "y": y})
    linear = [{"source": "x", "lag": 1, "target": "x"}, {"source": "y", "lag": 1, "target": "y"}]
    found = cg.as_set(cg.even_links(frame, linear))
    assert ("x", 1, "y") in found and all(link["kind"] == "even" for link in cg.even_links(frame, linear))
    chain, _truth = _chain(seed=3)
    parents = [{"source": s, "lag": lag, "target": t} for s, lag, t in [("x", 1, "x"), ("x", 1, "y"), ("y", 2, "z")]]
    assert len(cg.even_links(chain, parents)) <= 1


def test_worlds_are_stable_and_know_their_graph():
    world = worlds.make("nonlinear", 5, 300, seed=11)
    assert world["frame"].shape == (300, 5) and np.isfinite(world["frame"].to_numpy()).all()
    assert all((f"x{i}", 1, f"x{i}") in world["truth"] for i in range(5))
    assert set(world["function_of"]) == world["truth"]


@pytest.mark.skipif(not is_available(), reason="tigramite is not installed")
def test_pcmci_recovers_the_chain_without_the_indirect_link():
    frame, truth = _chain()
    found = cg.as_set(cg.discover_graph(frame))
    assert truth <= found and ("x", 3, "z") not in found
