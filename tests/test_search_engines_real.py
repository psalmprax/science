"""Behavioural tests for the beam, ensemble and NGTS search engines."""
import json

import numpy as np
import pandas as pd
import pytest

from apedp.advanced_search import HierarchicalSearch, NGTSSearch
from apedp.ensemble_search import EnsembleSearcher
from apedp.feature_selection import beam_search_expressions


def test_beam_is_deterministic_and_bounded():
    a = beam_search_expressions(["x", "y"], depth=2, width=8, seed=1)
    b = beam_search_expressions(["x", "y"], depth=2, width=8, seed=99)
    assert a == b
    assert len(a) == 20  # 4 atoms + 8 + 8


def test_beam_prefers_multi_variable_expressions():
    out = beam_search_expressions(["x", "y"], depth=1, width=4)
    level1 = out[4:]
    assert level1, out
    assert any("x" in e and "y" in e for e in level1)


def test_validator_scores_follow_measured_error_gap():
    from apedp.advanced_search import ValidationPipeline

    v = ValidationPipeline()
    v._rust = None  # exercise the Python formulas
    close = v.validate([0.1, 0.1, 0.1], 0.1)
    far = v.validate([0.1, 0.1, 0.1], 2.0)
    assert close["generalization_gap"] < 1e-12
    assert close["generalization_score"] > 1 - 1e-12
    assert far["generalization_gap"] > 1.0
    assert far["generalization_score"] < close["generalization_score"]
    assert far["ood_score"] < close["ood_score"]


def test_search_skips_candidates_without_measured_error():
    hs = HierarchicalSearch()
    out = hs.search(np.arange(20.0).reshape(-1, 1), np.arange(20.0), budget=30, variable_names=["x0"])
    assert out, "search should return validated candidates when data is given"
    assert all("r2" in c for c in out)
    assert all(c["overall_score"] == c["validation"]["overall"] for c in out)


def test_ensemble_rejects_unknown_member():
    frame = pd.DataFrame({"x": np.arange(40.0), "target": np.arange(40.0)})
    s = EnsembleSearcher(proposers=["nope"])
    out = s.search_ensemble(frame, ["x"], "target", budget_total=10)
    assert out["results"]["nope"]["status"] == "error"
    assert out["best"] is None


def _python_ngts():
    s = NGTSSearch("x0")
    s._rust = None  # exercise the Python UCT path regardless of the kernel
    return s


def test_rust_ngts_reward_callable_matches_python_loop():
    apedp_kernel = pytest.importorskip("apedp_kernel", reason="rust kernel not built")
    rewards = {"x0": 0.1, "x0**2": 0.9, "sin(x0)": 0.3}

    def reward(expr):
        return rewards.get(expr, 0.0)

    rs = apedp_kernel.NGTSSearch("x0")
    rs.set_config(1.414, 200, 10)
    rs.expand(0, json.dumps([[e, 1.0] for e in rewards]))
    assert rs.run_simulations(200, reward) == 200
    assert rs.best_expression() == "x0**2"


def test_ngts_without_reward_runs_no_simulations():
    s = _python_ngts()
    s.set_config(num_simulations=10)
    assert s.run_simulations(10) == 0


def test_hierarchical_mid_phase_finds_signal_expression():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 3.0, 120)
    y = x ** 2  # reachable from the mid-phase expansion set (x0**2)
    hs = HierarchicalSearch()
    seeds = [{"expression": "x0"}]
    cands = hs._mid_search(seeds, ["x0"], budget=60, data=[x], target=y)
    assert cands, "mid phase must return a candidate when a reward exists"
    assert cands[0]["expression"] == "x0**2"
    assert cands[0]["r2"] > 0.99


def test_hierarchical_mid_phase_reaches_scaled_form():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.5, 3.0, 120)
    y = 2.0 * x
    hs = HierarchicalSearch()
    cands = hs._mid_search([{"expression": "x0"}], ["x0"], budget=60, data=[x], target=y)
    assert cands and cands[0]["r2"] > 0.99


def test_hierarchical_fine_phase_scores_candidates_on_data():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.5, 3.0, 120)
    y = 2.0 * x
    hs = HierarchicalSearch()
    seeds = [{"expression": "x0*2", "complexity": 1}]
    out = hs._fine_search(seeds, ["x0"], budget=12, data=[x], target=y)
    assert out, "fine phase must return scored candidates when data exists"
    for cand in out:
        assert np.isfinite(cand["r2"])
    assert max(c["r2"] for c in out) > 0.99


def test_hierarchical_fine_phase_without_data_returns_nothing():
    hs = HierarchicalSearch()
    assert hs._fine_search([{"expression": "x0"}], ["x0"], budget=12) == []


def test_hierarchical_mid_phase_without_data_returns_nothing():
    hs = HierarchicalSearch()
    assert hs._mid_search([{"expression": "x0"}], ["x0"], budget=20) == []


def test_ngts_uct_prefers_rewarded_child():
    s = _python_ngts()
    s.reward_fn = lambda expr: 1.0 if expr == "good" else 0.0
    s.expand(0, [("bad", 0.5), ("good", 0.5)])
    done = s.run_simulations(50)
    assert done == 50
    assert s.best_expression() == "good"
