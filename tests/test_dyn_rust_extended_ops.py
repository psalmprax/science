"""Extended misc ops (neg/log/sqrt/abs) in enumeration and native GA."""
from __future__ import annotations

import json

import numpy as np

import apedp_kernel as ak
from apedp.evolutionary_search import SearchConfig, evolve


def _enum(x, y, extended):
    return json.loads(ak.rime_enumerate_search_json(
        json.dumps([np.asarray(x, dtype=float).tolist()]),
        json.dumps(np.asarray(y, dtype=float).tolist()),
        json.dumps(["v"]), False, False, False, extended, False, 2))


def test_enumeration_extended_ops_recover_each_form():
    v = np.linspace(0.5, 3.5, 40)
    cases = {"log": (v, np.log(v)), "sqrt": (v, np.sqrt(v)),
             "abs": (v - 2.0, np.abs(v - 2.0)), "neg": (v, -v)}
    for name, (x, y) in cases.items():
        res = _enum(x, y, True)
        assert len(res) > 0
        assert res[0]["r2"] > 0.999, (name, res[0])


def test_enumeration_extended_ops_gated_off_by_default():
    v = np.linspace(0.5, 3.5, 40)
    res = _enum(v, np.log(v), False)
    for c in res:
        assert "log(" not in c["formula"] and "sqrt(" not in c["formula"]


def test_native_ga_discovers_abs_and_log_with_misc_enabled():
    x = np.linspace(-3, 3, 200)
    r_abs = evolve({"x": x}, np.abs(x), config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    assert r_abs.best_expression is not None
    assert "abs" in r_abs.best_expression, r_abs.best_expression

    u = np.linspace(0.5, 4.0, 100)
    r_log = evolve({"u": u}, np.log(u), config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    assert r_log.best_expression is not None
    assert "log" in r_log.best_expression, r_log.best_expression


def test_native_ga_misc_opt_out_still_runs():
    x = np.linspace(-3, 3, 200)
    r = evolve({"x": x}, 3.0 * x ** 2 + 2.0, config=SearchConfig(
        use_native_ga=True, islands=4, island_size=50,
        generations=20, seed=42, allow_misc=False))
    mses = [p["train_mse"] for p in r.pareto_front]
    assert mses and min(mses) < 1e-6
