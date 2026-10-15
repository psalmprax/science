"""Exponential generation: RIME-AD rational pairing and native GA.

Covers the Planck grammar gap: `exp` nodes are now generatable in the
shared-DAG enumerator (rational-pairing stage) and in native GA trees.
"""
from __future__ import annotations

import json

import numpy as np

import apedp_kernel as ak
from apedp.evolutionary_search import SearchConfig, evolve
from apedp.rust_bridge import RustKernelBridge


def test_rational_pairs_recover_planck_textbook_form():
    # u(v) = v^3 / (e^v - 1): needs Div(v^3, Sub(Exp(v), 1)), which plain
    # depth-bounded enumeration cannot assemble (no binary ops at depth 3).
    v = np.linspace(0.1, 6.0, 60)
    u = (v ** 3) / (np.exp(v) - 1.0)
    res = json.loads(ak.rime_rational_search_json(
        json.dumps([v.tolist()]), json.dumps(u.tolist()),
        json.dumps(["v"]), False, False, False, False, 2, 3))
    assert len(res) > 0
    best = res[0]
    assert best["r2"] > 0.999, f"R2={best['r2']} formula={best['formula']}"
    assert "exp" in best["formula"], best["formula"]


def test_native_ga_with_exp_available_still_misses_planck_at_small_budget():
    # Grammar access is not discovery: even with exp generatable, 15
    # generations x 60 individuals cannot assemble v^3/(e^v-1) here.
    v_tr = np.linspace(0.1, 6.0, 60)
    v_ho = np.linspace(6.1, 10.0, 40)
    u_tr = (v_tr ** 3) / (np.exp(v_tr) - 1.0)
    u_ho = (v_ho ** 3) / (np.exp(v_ho) - 1.0)
    ga = evolve({"v": v_tr}, u_tr, config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42, allow_exp=True))
    assert ga.best_expression
    pred = RustKernelBridge().evaluate_expression(
        ga.best_expression, ["v"], [v_ho])
    ga_r2 = 1.0 - float(np.mean((u_ho - pred) ** 2)) / float(np.var(u_ho))
    assert ga_r2 < 0.9, f"GA unexpectedly solved Planck: R2={ga_r2}"
